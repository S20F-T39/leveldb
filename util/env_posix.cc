// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <map>
#include <string>

#include "leveldb/env.h"
#include "leveldb/slice.h"
#include "leveldb/status.h"
#include "port/port.h"
#include "port/thread_annotations.h"
#include "util/env_posix_test_helper.h"
#include "util/posix_logger.h"

#include <libzbc/zbc.h>

#define MAX_ZBC_BUFFER_SIZE 8388608

namespace leveldb {

namespace {

//Zone Mapping Table
std::map<std::string, struct zbc_zone*> map_table;
std::map<std::string, size_t> zone_size_map;
std::map<std::string, struct zbc_device*> path_dev_table;

// Set by EnvPosixTestHelper::SetReadOnlyMMapLimit() and MaxOpenFiles().
int g_open_read_only_file_limit = -1;

// Up to 1000 mmap regions for 64-bit binaries; none for 32-bit.
constexpr const int kDefaultMmapLimit = (sizeof(void*) >= 8) ? 1000 : 0;

// Can be set using EnvPosixTestHelper::SetReadOnlyMMapLimit().
int g_mmap_limit = kDefaultMmapLimit;

// Common flags defined for all posix open operations
#if defined(HAVE_O_CLOEXEC)
constexpr const int kOpenBaseFlags = O_CLOEXEC;
#else
constexpr const int kOpenBaseFlags = 0;
#endif  // defined(HAVE_O_CLOEXEC)

constexpr const size_t kWritableFileBufferSize = 65536;

Status PosixError(const std::string& context, int error_number) {
  if (error_number == ENOENT) {
    return Status::NotFound(context, std::strerror(error_number));
  } else {
    return Status::IOError(context, std::strerror(error_number));
  }
}

// Helper class to limit resource usage to avoid exhaustion.
// Currently used to limit read-only file descriptors and mmap file usage
// so that we do not run out of file descriptors or virtual memory, or run into
// kernel performance problems for very large databases.
class Limiter {
 public:
  // Limit maximum number of resources to |max_acquires|.
  Limiter(int max_acquires) : acquires_allowed_(max_acquires) {}

  Limiter(const Limiter&) = delete;
  Limiter operator=(const Limiter&) = delete;

  // If another resource is available, acquire it and return true.
  // Else return false.
  bool Acquire() {
    int old_acquires_allowed =
        acquires_allowed_.fetch_sub(1, std::memory_order_relaxed);

    if (old_acquires_allowed > 0) return true;

    acquires_allowed_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  // Release a resource acquired by a previous call to Acquire() that returned
  // true.
  void Release() { acquires_allowed_.fetch_add(1, std::memory_order_relaxed); }

 private:
  // The number of available resources.
  //
  // This is a counter and is not tied to the invariants of any other class, so
  // it can be operated on safely using std::memory_order_relaxed.
  std::atomic<int> acquires_allowed_;
};

// Implements sequential read access in a file using read().
//
// Instances of this class are thread-friendly but not thread-safe, as required
// by the SequentialFile API.
class PosixSequentialFile final : public SequentialFile {
 public:
  // ZNS Device 와 target_zone 을 인수로 Class 생성
  PosixSequentialFile(struct zbc_device *dev, struct zbc_zone *target_zone, std::string filename)
      : dev_(dev),
        target_zone_(target_zone),
        filename_(std::move(filename)) {}

  ~PosixSequentialFile() override {}

  Status Read(size_t n, Slice* result, char* scratch) override {
    // sector_count 지정. 512B 보다 작을 때, 1로 주어 Read 보장.
    ssize_t sector_count = n >> 9;
    if (sector_count < 1) sector_count = 1;

    size_t target_size = zone_size_map[filename_];

    // zbc_pread: 해당 zone 시작부터 sequential 하게 읽어와 scratch 에 넣어줌.
    size_t read_size = zbc_pread(dev_, scratch, sector_count, target_zone_->zbz_start);
    if (read_size < 0) {  // Read error. 
      return PosixError("PosixSequentialFile: zbc_pread failed.\n", errno);
    }
    
    *result = Slice(scratch, target_size);

    // CURRENT File 일 때, size mapping table 초기화
    if (!strcmp(filename_.c_str(), "/tmp/leveldbtest-0/dbbench/CURRENT")) 
      zone_size_map.erase(filename_);

    return Status::OK();
  }

  Status Skip(uint64_t n) override {
    return Status::OK();
  }

 private:
  struct zbc_zone *target_zone_;
  struct zbc_device *dev_;
  std::string filename_;
};

// Implements random read access in a file using pread().
//
// Instances of this class are thread-safe, as required by the RandomAccessFile
// API. Instances are immutable and Read() only calls thread-safe library
// functions.
class PosixRandomAccessFile final : public RandomAccessFile {
 public:
  // The new instance takes ownership of |fd|. |fd_limiter| must outlive this
  // instance, and will be used to determine if .
  PosixRandomAccessFile(struct zbc_device *dev, struct zbc_zone *target_zone, std::string filename)
      : dev_(dev),
        target_zone_(target_zone),
        filename_(std::move(filename)) {}

  ~PosixRandomAccessFile() override {}

  // RandomAccessFile -> offset 이 오면, 해당 offset 부터 읽기 시작.
  Status Read(uint64_t offset, size_t n, Slice* result, char* scratch) const override {
    uint64_t sector_start;
    uint64_t sector_offset;

    size_t file_size = zone_size_map[filename_];

    // sector_count 지정. 512B 보다 작을 때, 1로 주어 Read 보장.
    // 512B 로 나누었을 때, 해당 값보다 1 더해 주어야 함.
    size_t sector_count = (file_size >> 9) + 1;
    if (sector_count < 1) {
      sector_count = 1;
    }

    char *tmp_buffer = nullptr;

    // tmp_buffer 의 size 를 sector_count 의 512B 배수로 맞춰 줌.
    size_t tmp_buf_size = sector_count << 9;
    size_t ret = posix_memalign((void **) &tmp_buffer, sysconf(_SC_PAGESIZE), tmp_buf_size);
    if (ret != 0) {
      fprintf(stderr, "No memory for I/O buffer (%zu B)\n", tmp_buf_size);
      return PosixError("RandomAccessFile::Read Failed", errno);
    }
    // memset(tmp_buffer, 0, tmp_buf_size);

    // 계산한 sector 갯수만큼 처음부터 읽어서 tmp_buffer 에 Read
    printf("Start Sector: %llu\n", target_zone_->zbz_start);
    size_t read_size = zbc_pread(dev_, tmp_buffer, sector_count, target_zone_->zbz_start);
    if (read_size < 0) {
      return PosixError("PosixRandomAccessFile::Read Failed", errno);
    }

    // for (uint64_t i = offset; i < file_size; i++) {
    //   printf("%x", tmp_buffer[i]);
    // }
    // printf("\n");

    *result = Slice(scratch, 1);

    return Status::OK();
  }

 private:
  struct zbc_zone *target_zone_;
  struct zbc_device *dev_;
  std::string filename_;
};

// Implements random read access in a file using mmap().
//
// Instances of this class are thread-safe, as required by the RandomAccessFile
// API. Instances are immutable and Read() only calls thread-safe library
// functions.
class PosixMmapReadableFile final : public RandomAccessFile {
 public:
  // mmap_base[0, length-1] points to the memory-mapped contents of the file. It
  // must be the result of a successful call to mmap(). This instances takes
  // over the ownership of the region.
  //
  // |mmap_limiter| must outlive this instance. The caller must have already
  // aquired the right to use one mmap region, which will be released when this
  // instance is destroyed.
  PosixMmapReadableFile(std::string filename, char* mmap_base, size_t length,
                        Limiter* mmap_limiter)
      : mmap_base_(mmap_base),
        length_(length),
        mmap_limiter_(mmap_limiter),
        filename_(std::move(filename)) {}

  ~PosixMmapReadableFile() override {
    ::munmap(static_cast<void*>(mmap_base_), length_);
    mmap_limiter_->Release();
  }

  Status Read(uint64_t offset, size_t n, Slice* result,
              char* scratch) const override {
    if (offset + n > length_) {
      *result = Slice();
      return PosixError(filename_, EINVAL);
    }

    *result = Slice(mmap_base_ + offset, n);
    return Status::OK();
  }

 private:
  char* const mmap_base_;
  const size_t length_;
  Limiter* const mmap_limiter_;
  const std::string filename_;
};

class PosixWritableFile final : public WritableFile {
 public:
  PosixWritableFile(zbc_device *dev, zbc_zone *target_zone, std::string filename)
      : dev_(dev),
        target_zone_(target_zone),
        pos_(0),
        is_manifest_(IsManifest(filename)),
        filename_(std::move(filename)),
        dirname_(Dirname(filename_)) {}

  ~PosixWritableFile() override {
    Close();
  }

  Status Append(const Slice& data) override {
    size_t write_size = data.size();
    const char* write_data = data.data();

    // Fit as much as possible into buffer.
    size_t copy_size = std::min(write_size, kWritableFileBufferSize - pos_);
    std::memcpy(buf_ + pos_, write_data, copy_size);
    write_data += copy_size;
    write_size -= copy_size;
    pos_ += copy_size;
    if (write_size == 0) {
      return Status::OK();
    }

    // Can't fit in buffer, so need to do at least one write.
    Status status = FlushBuffer();
    if (!status.ok()) {
      return status;
    }

    // Small writes go to buffer, large writes are written directly.
    if (write_size < kWritableFileBufferSize) {
      std::memcpy(buf_, write_data, write_size);
      pos_ = write_size;
      return Status::OK();
    }
    return WriteUnbuffered(write_data, write_size);
  }

  Status Close() override {
    Status status = FlushBuffer();
    return status;
  }

  Status Flush() override { return FlushBuffer(); }

  Status Sync() override {
    // Ensure new files referred to by the manifest are in the filesystem.
    //
    // This needs to happen before the manifest file is flushed to disk, to
    // avoid crashing in a state where the manifest refers to files that are not
    // yet on disk.
    // Status status = SyncDirIfManifest();
    // if (!status.ok()) {
    //   return status;
    // }

    Status status = FlushBuffer();
    if (!status.ok()) {
      return status;
    }

    // return SyncFd(zone_number, filename_);
    return status;
  }

 private:
  Status FlushBuffer() {
    Status status = WriteUnbuffered(buf_, pos_);
    pos_ = 0;
    return status;
  }

  Status WriteUnbuffered(const char* data, size_t size) {
    uint64_t sector_start;
    size_t sector_count;

    if (size == 0) sector_count = 0;
    else {
      if (size % 512 == 0) sector_count = size >> 9;
      else {
        if ((size >> 9) < 1) sector_count = 1;
        else sector_count = (size >> 9) + 1;
      }
    }

    zone_size_map[filename_] += size;
    
    // 쓰기 시작할 sector_start 에 대해서 정해 줌.
    // 해당 target_zone 의 wp 부터 순차적으로 write
    sector_start = zbc_zone_wp(map_table[filename_]);
    size_t write_result = zbc_pwrite(dev_, data, sector_count, sector_start);
    if (write_result < 0) {
      return PosixError("PosixWritableFile::WriteUnbuffered Failed", errno);
    }
    printf("Write Result: %llu, wp: %llu\n", write_result, sector_start);
    
    return Status::OK();
  }

  Status SyncDirIfManifest() {
    Status status;
    if (!is_manifest_) {
      return status;
    }

    // int fd = ::open(dirname_.c_str(), O_RDONLY | kOpenBaseFlags);
    // if (fd < 0) {
    //   status = PosixError(dirname_, errno);
    // } else {
    //   status = SyncFd(fd, dirname_);
    //   ::close(fd);
    // }
    return Status::OK();
  }

  // Ensures that all the caches associated with the given file descriptor's
  // data are flushed all the way to durable media, and can withstand power
  // failures.
  //
  // The path argument is only used to populate the description string in the
  // returned Status if an error occurs.
  static Status SyncFd(int fd, const std::string& fd_path) {
#if HAVE_FULLFSYNC
    // On macOS and iOS, fsync() doesn't guarantee durability past power
    // failures. fcntl(F_FULLFSYNC) is required for that purpose. Some
    // filesystems don't support fcntl(F_FULLFSYNC), and require a fallback to
    // fsync().
    if (::fcntl(fd, F_FULLFSYNC) == 0) {
      return Status::OK();
    }
#endif  // HAVE_FULLFSYNC

#if HAVE_FDATASYNC
    bool sync_success = ::fdatasync(fd) == 0;
#else
    bool sync_success = ::fsync(fd) == 0;
#endif  // HAVE_FDATASYNC

    if (sync_success) {
      return Status::OK();
    }
    return PosixError(fd_path, errno);
  }

  // Returns the directory name in a path pointing to a file.
  //
  // Returns "." if the path does not contain any directory separator.
  static std::string Dirname(const std::string& filename) {
    std::string::size_type separator_pos = filename.rfind('/');
    if (separator_pos == std::string::npos) {
      return std::string(".");
    }
    // The filename component should not contain a path separator. If it does,
    // the splitting was done incorrectly.
    assert(filename.find('/', separator_pos + 1) == std::string::npos);

    return filename.substr(0, separator_pos);
  }

  // Extracts the file name from a path pointing to a file.
  //
  // The returned Slice points to |filename|'s data buffer, so it is only valid
  // while |filename| is alive and unchanged.
  static Slice Basename(const std::string& filename) {
    std::string::size_type separator_pos = filename.rfind('/');
    if (separator_pos == std::string::npos) {
      return Slice(filename);
    }
    // The filename component should not contain a path separator. If it does,
    // the splitting was done incorrectly.
    assert(filename.find('/', separator_pos + 1) == std::string::npos);

    return Slice(filename.data() + separator_pos + 1,
                 filename.length() - separator_pos - 1);
  }

  // True if the given file is a manifest file.
  static bool IsManifest(const std::string& filename) {
    return Basename(filename).starts_with("MANIFEST");
  }

  size_t pos_;
  char buf_[kWritableFileBufferSize];

  const bool is_manifest_;
  const std::string filename_;
  const std::string dirname_;

  // zbc device
  struct zbc_device *dev_;
  struct zbc_zone *target_zone_;
};

int LockOrUnlock(int ret, bool lock) {
  return ret;
}

// Instances are thread-safe because they are immutable.
class PosixFileLock : public FileLock {
 public:
  PosixFileLock(std::string filename, zbc_device *dev)
      : filename_(std::move(filename)),
        dev_(dev) {}

  const std::string& filename() const { return filename_; }
  struct zbc_device* dev() const { return dev_; }

 private:
  struct zbc_device *dev_;
  const std::string filename_;
};

// Tracks the files locked by PosixEnv::LockFile().
//
// We maintain a separate set instead of relying on fcntl(F_SETLK) because
// fcntl(F_SETLK) does not provide any protection against multiple uses from the
// same process.
//
// Instances are thread-safe because all member data is guarded by a mutex.
class PosixLockTable {
 public:
  bool Insert(const std::string& fname) LOCKS_EXCLUDED(mu_) {
    mu_.Lock();
    bool succeeded = locked_files_.insert(fname).second;
    mu_.Unlock();
    return succeeded;
  }
  void Remove(const std::string& fname) LOCKS_EXCLUDED(mu_) {
    mu_.Lock();
    locked_files_.erase(fname);
    mu_.Unlock();
  }

 private:
  port::Mutex mu_;
  std::set<std::string> locked_files_ GUARDED_BY(mu_);
};

class PosixEnv : public Env {
 public:
  PosixEnv();
  ~PosixEnv() override {
    static const char msg[] =
        "PosixEnv singleton destroyed. Unsupported behavior!\n";
    std::fwrite(msg, 1, sizeof(msg), stderr);
    std::abort();
  }

  // Device Open Function
  size_t OpenZonedDevice(std::string path) {
    size_t ret;
    struct zbc_device *dev = nullptr;

    // path 와 연결된 device 없을 때. return 0
    // path 와 연결된 device 있을 때. return 1
    if (path_dev_table.find(path) == path_dev_table.end()) {
      ret = zbc_open(path.c_str(), O_RDWR, &dev);
      if (ret != 0) {
        if (ret == -ENODEV) 
          fprintf(stderr, "Open %s failed (not a zoned block device)\n", path.c_str());
        else 
          fprintf(stderr, "Open %s failed (%s)\n", path.c_str(), strerror(-ret));
        return ret;
      }
      path_dev_table[path] = dev;
      return 0;
    }
    return 1;
  }

  // Zone Mapping Function
  ssize_t FindZoneForFilename(zbc_device *dev, zbc_zone **target_zone) {
    ssize_t ret;
    struct zbc_zone *result_zones = nullptr;
    struct zbc_zone *imp_open_zones = nullptr;
    unsigned int nr_result_zones;
    unsigned int nr_imp_open_zones;

    // 1. Empty Zone 가져 옴.
    ret = zbc_list_zones(dev, 0, ZBC_RO_EMPTY, &result_zones, &nr_result_zones);
    if (ret != 0) {
      fprintf(stderr, "FindZoneForFilename: zbc_list_zones(EMPTY) failed\n");
      return ret;
    }

    // Empty Zone 있다면 target_zone = 첫번째 Empty Zone
    if (nr_result_zones != 0) {
      *target_zone = &result_zones[0];
      return ret;
    }
    
    return ret;
  }

  Status NewSequentialFile(const std::string& filename, SequentialFile** result) override {
    std::string path = "/dev/sda";

    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    OpenZonedDevice(path);

    dev = path_dev_table[path];
    target_zone = map_table[filename];

    *result = new PosixSequentialFile(dev, target_zone, filename);

    return Status::OK();
  }

  Status NewRandomAccessFile(const std::string& filename, RandomAccessFile** result) override {
    std::string path = "/dev/sda";

    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    OpenZonedDevice(path);

    dev = path_dev_table[path];
    target_zone = map_table[filename];

    *result = new PosixRandomAccessFile(dev, target_zone, filename);

    return Status::OK();
  }

  Status NewWritableFile(const std::string& filename, WritableFile** result) override {
    std::string path = "/dev/sda";
    size_t ret;

    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    OpenZonedDevice(path);
    dev = path_dev_table[path];

    std::string current_file_name;

    // dbtmp File 에 대해서.
    if (filename.find("dbtmp", 0) != std::string::npos)
      current_file_name = "/tmp/leveldbtest-0/dbbench/CURRENT";
    else current_file_name = filename;
    
    if (map_table.find(current_file_name) != map_table.end()) {
      target_zone = map_table[current_file_name];
    } else {
      ret = FindZoneForFilename(dev, &target_zone);
      if (ret != 0) {
        fprintf(stderr, "NewWritableFile: FindZoneForFilename failed\n");
        return PosixError("NewWritableFile: FindZoneForFilename", errno);
      }
      // target_zone 함수 통해서 받아 왔을 때, filename 과 매핑.
      leveldb::map_table[current_file_name] = target_zone;
      zbc_open_zone(dev, target_zone->zbz_start, 0);
    }

    *result = new PosixWritableFile(dev, target_zone, current_file_name);

    return Status::OK();
  }

  Status NewAppendableFile(const std::string& filename, WritableFile** result) override {
    std::string path = "/dev/sda";
    size_t ret;
    
    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    OpenZonedDevice(path);
    dev = path_dev_table[path];

    std::string current_file_name;

    // dbtmp File 에 대해서.
    if (filename.find("dbtmp", 0) != std::string::npos)
      current_file_name = "/tmp/leveldbtest-0/dbbench/CURRENT";
    else current_file_name = filename;
    
    if (map_table.find(current_file_name) != map_table.end()) {
      target_zone = map_table[current_file_name];
    } else {
      ret = FindZoneForFilename(dev, &target_zone);
      if (ret != 0) {
        fprintf(stderr, "NewAppendableFile: FindZoneForFilename failed\n");
        return PosixError("NewAppendableFile: FindZoneForFilename", errno);
      }
    }

    // target_zone 함수 통해서 받아 왔을 때, filename 과 매핑.
    leveldb::map_table[current_file_name] = target_zone;
    zbc_open_zone(dev, target_zone->zbz_start, 0);

    *result = new PosixWritableFile(dev, target_zone, current_file_name);

    return Status::OK();
  }

  bool FileExists(const std::string& filename) override {
    // map 에서 key 로 가지고 있는 filename 이라면 true return
    // 그것이 아니라면 false
    return leveldb::map_table.find(filename) != leveldb::map_table.end();
  }

  Status GetChildren(const std::string& directory_path,
                     std::vector<std::string>* result) override {
    result->clear();
    return Status::OK();
  }

  Status RemoveFile(const std::string& filename) override {
    std::string path = "/dev/sda";
    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    // 해당 filename 과 mapping 된 target_zone 받아옴.
    OpenZonedDevice(path);
    dev = path_dev_table[path];
    target_zone = map_table[filename];

    // 해당 target_zone reset
    zbc_reset_zone(dev, target_zone->zbz_start, 0);

    // zone reset 후, mapping_table 에서도 지워줘야 함.
    free(map_table[filename]);
    map_table.erase(filename);

    printf("RemoveFile \"%s\"\n", filename.c_str());

    return Status::OK();
  }

  Status CreateDir(const std::string& dirname) override {
    std::string path = "/dev/sda";
    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    OpenZonedDevice(path);
    dev = path_dev_table[path];
    
    // 해당 dirname 과 mapping 시켜 줄 target_zone 받아와서, 매핑.
    FindZoneForFilename(dev, &target_zone);
    map_table[dirname] = target_zone;

    return Status::OK();
  }

  Status RemoveDir(const std::string& dirname) override {
    std::string path = "/dev/sda";
    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    OpenZonedDevice(path);
    dev = path_dev_table[path];

    // 해당 dirname 과 mapping 된 target_zone 받아옴.
    target_zone = map_table[dirname];

    // 첫 시작과 함께 하위 디렉터리 삭제하기 때문에,
    // filename 과 매핑된 zone 없음. 모든 zone reset
    if (target_zone == nullptr) {
      zbc_reset_zone(dev, 0, ZBC_OP_ALL_ZONES);
      return Status::OK();
    }

    // 해당 target_zone reset
    zbc_reset_zone(dev, target_zone->zbz_start, 0);

    // zone reset 후, mapping_table 에서도 지워줘야 함.
    free(map_table[dirname]);
    map_table.erase(dirname);

    return Status::OK();
  }

  Status GetFileSize(const std::string& filename, uint64_t* size) override {
    size_t file_size;

    // filename 과 매핑된 사이즈
    file_size = zone_size_map[filename];
    if (file_size == 0) {
      *size = 0;
      return PosixError("GetFileSize: no target zone for " + filename, errno);
    } 

    *size = file_size;

    return Status::OK();
  }

  Status RenameFile(const std::string& from, const std::string& to) override {
    return Status::OK();
  }

  Status LockFile(const std::string& filename, FileLock** lock) override {
    *lock = nullptr;

    std::string path = "/dev/sda";
    size_t ret;
    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    OpenZonedDevice(path);
    dev = path_dev_table[path];

    FindZoneForFilename(dev, &target_zone);
    map_table[filename] = target_zone;

    if (!locks_.Insert(filename)) {
      return Status::IOError("lock " + filename, "already held by process");
    }

    if (LockOrUnlock(ret, true) == -1) {
      // ZBC Device Close.
      int lock_errno = errno;
      locks_.Remove(filename);
      return PosixError("lock " + filename, lock_errno);
    }

    *lock = new PosixFileLock(filename, dev);

    return Status::OK();
  }

  Status UnlockFile(FileLock* lock) override {
    PosixFileLock* posix_file_lock = static_cast<PosixFileLock*>(lock);
    if (LockOrUnlock(0, false) == -1) {
      return PosixError("unlock " + posix_file_lock->filename(), errno);
    }
    locks_.Remove(posix_file_lock->filename());
    delete posix_file_lock;
    return Status::OK();
  }

  void Schedule(void (*background_work_function)(void* background_work_arg),
                void* background_work_arg) override;

  void StartThread(void (*thread_main)(void* thread_main_arg),
                   void* thread_main_arg) override {
    std::thread new_thread(thread_main, thread_main_arg);
    new_thread.detach();
  }

  Status GetTestDirectory(std::string* result) override {
    const char* env = std::getenv("TEST_TMPDIR");
    if (env && env[0] != '\0') {
      *result = env;
    } else {
      char buf[100];
      std::snprintf(buf, sizeof(buf), "/tmp/leveldbtest-%d",
                    static_cast<int>(::geteuid())); //libzbc
      *result = buf;
    }

    // The CreateDir status is ignored because the directory may already exist.
    // libzbc 사용 시 Dir 필요가 없으므로, CreateDir 수행하지 않도록 수정.
    CreateDir(*result);

    return Status::OK();
  }

  Status NewLogger(const std::string& filename, Logger** result) override {
    std::string path = "/dev/sda";
    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    OpenZonedDevice(path);
    dev = path_dev_table[path];

    // posix open 대신 map_table 에 매핑 진행
    FindZoneForFilename(dev, &target_zone);
    map_table[filename] = target_zone;

    *result = new PosixLogger(dev, target_zone);
    return Status::OK();
  }

  uint64_t NowMicros() override {
    static constexpr uint64_t kUsecondsPerSecond = 1000000;
    struct ::timeval tv;
    ::gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * kUsecondsPerSecond + tv.tv_usec;
  }

  void SleepForMicroseconds(int micros) override {
    std::this_thread::sleep_for(std::chrono::microseconds(micros));
  }

 private:
  void BackgroundThreadMain();

  static void BackgroundThreadEntryPoint(PosixEnv* env) {
    env->BackgroundThreadMain();
  }

  // Stores the work item data in a Schedule() call.
  //
  // Instances are constructed on the thread calling Schedule() and used on the
  // background thread.
  //
  // This structure is thread-safe beacuse it is immutable.
  struct BackgroundWorkItem {
    explicit BackgroundWorkItem(void (*function)(void* arg), void* arg)
        : function(function), arg(arg) {}

    void (*const function)(void*);
    void* const arg;
  };

  port::Mutex background_work_mutex_;
  port::CondVar background_work_cv_ GUARDED_BY(background_work_mutex_);
  bool started_background_thread_ GUARDED_BY(background_work_mutex_);

  std::queue<BackgroundWorkItem> background_work_queue_
      GUARDED_BY(background_work_mutex_);

  PosixLockTable locks_;  // Thread-safe.
  Limiter mmap_limiter_;  // Thread-safe.
  Limiter fd_limiter_;    // Thread-safe.
};

// Return the maximum number of concurrent mmaps.
int MaxMmaps() { return g_mmap_limit; }

// Return the maximum number of read-only files to keep open.
int MaxOpenFiles() {
  if (g_open_read_only_file_limit >= 0) {
    return g_open_read_only_file_limit;
  }
  struct ::rlimit rlim;
  if (::getrlimit(RLIMIT_NOFILE, &rlim)) {
    // getrlimit failed, fallback to hard-coded default.
    g_open_read_only_file_limit = 50;
  } else if (rlim.rlim_cur == RLIM_INFINITY) {
    g_open_read_only_file_limit = std::numeric_limits<int>::max();
  } else {
    // Allow use of 20% of available file descriptors for read-only files.
    g_open_read_only_file_limit = rlim.rlim_cur / 5;
  }
  return g_open_read_only_file_limit;
}

}  // namespace

PosixEnv::PosixEnv()
    : background_work_cv_(&background_work_mutex_),
      started_background_thread_(false),
      mmap_limiter_(MaxMmaps()),
      fd_limiter_(MaxOpenFiles()) {}

void PosixEnv::Schedule(
    void (*background_work_function)(void* background_work_arg),
    void* background_work_arg) {
  background_work_mutex_.Lock();

  // Start the background thread, if we haven't done so already.
  if (!started_background_thread_) {
    started_background_thread_ = true;
    std::thread background_thread(PosixEnv::BackgroundThreadEntryPoint, this);
    background_thread.detach();
  }

  // If the queue is empty, the background thread may be waiting for work.
  if (background_work_queue_.empty()) {
    background_work_cv_.Signal();
  }

  background_work_queue_.emplace(background_work_function, background_work_arg);
  background_work_mutex_.Unlock();
}

void PosixEnv::BackgroundThreadMain() {
  while (true) {
    background_work_mutex_.Lock();

    // Wait until there is work to be done.
    while (background_work_queue_.empty()) {
      background_work_cv_.Wait();
    }

    assert(!background_work_queue_.empty());
    auto background_work_function = background_work_queue_.front().function;
    void* background_work_arg = background_work_queue_.front().arg;
    background_work_queue_.pop();

    background_work_mutex_.Unlock();
    background_work_function(background_work_arg);
  }
}

namespace {

// Wraps an Env instance whose destructor is never created.
//
// Intended usage:
//   using PlatformSingletonEnv = SingletonEnv<PlatformEnv>;
//   void ConfigurePosixEnv(int param) {
//     PlatformSingletonEnv::AssertEnvNotInitialized();
//     // set global configuration flags.
//   }
//   Env* Env::Default() {
//     static PlatformSingletonEnv default_env;
//     return default_env.env();
//   }
template <typename EnvType>
class SingletonEnv {
 public:
  SingletonEnv() {
#if !defined(NDEBUG)
    env_initialized_.store(true, std::memory_order::memory_order_relaxed);
#endif  // !defined(NDEBUG)
    static_assert(sizeof(env_storage_) >= sizeof(EnvType),
                  "env_storage_ will not fit the Env");
    static_assert(alignof(decltype(env_storage_)) >= alignof(EnvType),
                  "env_storage_ does not meet the Env's alignment needs");
    new (&env_storage_) EnvType();
  }
  ~SingletonEnv() = default;

  SingletonEnv(const SingletonEnv&) = delete;
  SingletonEnv& operator=(const SingletonEnv&) = delete;

  Env* env() { return reinterpret_cast<Env*>(&env_storage_); }

  static void AssertEnvNotInitialized() {
#if !defined(NDEBUG)
    assert(!env_initialized_.load(std::memory_order::memory_order_relaxed));
#endif  // !defined(NDEBUG)
  }

 private:
  typename std::aligned_storage<sizeof(EnvType), alignof(EnvType)>::type
      env_storage_;
#if !defined(NDEBUG)
  static std::atomic<bool> env_initialized_;
#endif  // !defined(NDEBUG)
};

#if !defined(NDEBUG)
template <typename EnvType>
std::atomic<bool> SingletonEnv<EnvType>::env_initialized_;
#endif  // !defined(NDEBUG)

using PosixDefaultEnv = SingletonEnv<PosixEnv>;

}  // namespace

void EnvPosixTestHelper::SetReadOnlyFDLimit(int limit) {
  PosixDefaultEnv::AssertEnvNotInitialized();
  g_open_read_only_file_limit = limit;
}

void EnvPosixTestHelper::SetReadOnlyMMapLimit(int limit) {
  PosixDefaultEnv::AssertEnvNotInitialized();
  g_mmap_limit = limit;
}

Env* Env::Default() {
  static PosixDefaultEnv env_container;
  return env_container.env();
}

}  // namespace leveldb
