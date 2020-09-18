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

namespace leveldb {

namespace {

//Zone Mapping Table
std::map<std::string, struct zbc_zone*> map_table;

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
  PosixSequentialFile(struct zbc_device *dev, struct zbc_zone *target_zone)
      : dev_(dev),
        target_zone_(target_zone) {}

  ~PosixSequentialFile() override {
    printf("PosixSequentialFile Destroyed...\n");
    zbc_close(dev_);
  }

  Status Read(size_t n, Slice* result, char* scratch) override {
    Status status;
    while (true) {
      // zbc_pread
      ::ssize_t read_size = zbc_pread(dev_, scratch, n >> 9, target_zone_->zbz_write_pointer);
      if (read_size < 0) {  // Read error. 
        if (errno == EINTR) continue;  // Retry
        status = PosixError("PosixSequentialFile: zbc_pread failed.\n", errno);
        break;
      }
      printf("PosixSequentialFile::Read Complete...\n");
      printf("Read Result:: size: %llu / data: %s / counter: %llu / wp: %llu\n", 
      read_size, scratch, n >> 9, target_zone_->zbz_write_pointer);
      *result = Slice(scratch, read_size);
      break;
    }

    return status;
  }

  Status Skip(uint64_t n) override {
    return Status::OK();
  }

 private:
  struct zbc_zone *target_zone_;
  struct zbc_device *dev_;
  struct zbc_device_info info_ {};
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
  PosixRandomAccessFile(std::string filename, int fd, Limiter* fd_limiter)
      : has_permanent_fd_(fd_limiter->Acquire()),
        fd_(has_permanent_fd_ ? fd : -1),
        fd_limiter_(fd_limiter),
        filename_(std::move(filename)) {
    if (!has_permanent_fd_) {
      assert(fd_ == -1);
      ::close(fd);  // The file will be opened on every read.
    }
  }

  ~PosixRandomAccessFile() override {
    if (has_permanent_fd_) {
      assert(fd_ != -1);
      ::close(fd_);
      fd_limiter_->Release();
    }
  }


  Status Read(uint64_t offset, size_t n, Slice* result,
              char* scratch) const override {
    int fd = fd_;
    if (!has_permanent_fd_) {
      fd = ::open(filename_.c_str(), O_RDONLY | kOpenBaseFlags);
      if (fd < 0) {
        return PosixError(filename_, errno);
      }
    }

    assert(fd != -1);

    Status status;
    ssize_t read_size = ::pread(fd, scratch, n, static_cast<off_t>(offset));
    *result = Slice(scratch, (read_size < 0) ? 0 : read_size);
    if (read_size < 0) {
      // An error: return a non-ok status.
      status = PosixError(filename_, errno);
    }
    if (!has_permanent_fd_) {
      // Close the temporary file descriptor opened earlier.
      assert(fd != fd_);
      ::close(fd);
    }
    return status;
  }

 private:
  const bool has_permanent_fd_;  // If false, the file is opened on every read.
  const int fd_;                 // -1 if has_permanent_fd_ is false.
  Limiter* const fd_limiter_;
  const std::string filename_;
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

    zbc_close(dev_);

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
    // Variables for zbc zones
    ssize_t ret;
    ssize_t sector_count;
    long long zone_ofst = 0;
    long long sector_ofst;

    while (size > 0) {
      sector_count = size >> 9;
      sector_ofst = zbc_zone_start(target_zone_) + zone_ofst;

      ret = zbc_pwrite(dev_, (void *) data, sector_count, sector_ofst);
      if (ret < 0) {
        fprintf(stderr, "%s failed %zd (%s)\n", "zbc_pwrite", -ret, strerror(-ret));
        zbc_close(dev_);
        return PosixError(filename_, errno);
      }

      // LOCK File 같이 빈 File 들은 ret = 0.
      if (ret == 0) break;

      zone_ofst += ret;
      data += ret;
      size -= ret;
    }

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

  // Zone Mapping Function
  ssize_t FindZoneForFilename(zbc_device *dev, zbc_zone **target_zone) {
    ssize_t ret;
    struct zbc_zone *result_zones = nullptr;
    unsigned int nr_result_zones;

    // 1. Empty Zone 가져 옴.
    ret = zbc_list_zones(dev, 0, ZBC_RO_EMPTY, &result_zones, &nr_result_zones);
    if (ret != 0) {
      fprintf(stderr, "FindZoneForFilename: zbc_list_zones(EMPTY) failed\n");
      return ret;
    }
    
    // 2. Empty Zone 이 유효할 때, target_zone 을 첫 번째 Empty Zone 으로 지정.
    if (nr_result_zones != 0) {
      *target_zone = &result_zones[0];
      return ret;
    }

    // 3. Empty Zone 이 존재하지 않을 때,
    // target_zone 을 Closed Zone 중 하나로 매핑.
    // 먼저 closed zone 가져 옴.
    ret = zbc_list_zones(dev, 0, ZBC_RO_CLOSED, &result_zones, &nr_result_zones);
    if (ret != 0) {
      fprintf(stderr, "FindZoneForFilename: zbc_list_zones(CLOSED) failed\n");
      return ret;
    }

    // closed zone list 가져 왔다면, 첫 번째 closed zone reset.
    // 그 후에, closed zone 을 target_zone 으로 설정 하고 return.
    if (nr_result_zones != 0) {
      ret = zbc_reset_zone(dev, zbc_zone_start(&result_zones[0]), 0);
      if (ret == -EIO) {
        fprintf(stderr, "FindZoneForFilename: zbc_reset_zone failed\n");
        return ret;
      }
      *target_zone = &result_zones[0];
      return ret;
    }

    // 4. Closed Zone 또한 존재하지 않을 때,
    // target_zone 을 Imp_Open Zone 중 하나로 매핑.
    ret = zbc_list_zones(dev, 0, ZBC_RO_IMP_OPEN, &result_zones, &nr_result_zones);
    if (ret != 0) {
      fprintf(stderr, "FindZoneForFilename: zbc_list_zones(IMP_OPEN) failed\n");
      return ret;
    }

    // Imp_Open zone 까지 실패할 경우에는,
    // Error Return.
    if (nr_result_zones == 0) {
      fprintf(stderr, "FindZoneForFilename: Any Zone for mapping.\n");
      return ret;
    }

    ret = zbc_reset_zone(dev, result_zones[0].zbz_start, 0);
    if (ret == -EIO) {
      fprintf(stderr, "FindZoneForFilename: zbc_reset_zone failed\n");
      return ret;
    }

    fprintf(stderr, "FindZoneForFilename: final\n");
    *target_zone = &result_zones[0];
    return ret;
  }

  Status NewSequentialFile(const std::string& filename,
                           SequentialFile** result) override {
    
    std::string path = "/dev/sda";

    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    // Open Device
    ssize_t ret = zbc_open(path.c_str(), O_RDWR, &dev);
    if (ret != 0) {
      if (ret == -ENODEV) 
        fprintf(stderr, "Open %s failed (not a zoned block device)\n", path.c_str());
      else 
        fprintf(stderr, "Open %s failed (%s)\n", path.c_str(), strerror(-ret));
      return PosixError("NewSequentialFile: zbc_open", errno);
    }

    // filename 이 key 로 존재하지 않을 때.
    if (map_table.find(filename) == map_table.end()) {
      FindZoneForFilename(dev, &target_zone);
      map_table[filename] = target_zone;
    } else {
      // filename 사용하여 target_zone 가져 옴.
      target_zone = leveldb::map_table[filename];
    }
    
    *result = new PosixSequentialFile(dev, target_zone);

    return Status::OK();
  }

  Status NewRandomAccessFile(const std::string& filename,
                             RandomAccessFile** result) override {
    *result = nullptr;
    int fd = ::open(filename.c_str(), O_RDONLY | kOpenBaseFlags);
    if (fd < 0) {
      return PosixError(filename, errno);
    }

    if (!mmap_limiter_.Acquire()) {
      *result = new PosixRandomAccessFile(filename, fd, &fd_limiter_);
      return Status::OK();
    }

    uint64_t file_size;
    Status status = GetFileSize(filename, &file_size);
    if (status.ok()) {
      void* mmap_base =
          ::mmap(/*addr=*/nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
      if (mmap_base != MAP_FAILED) {
        *result = new PosixMmapReadableFile(filename,
                                            reinterpret_cast<char*>(mmap_base),
                                            file_size, &mmap_limiter_);
      } else {
        status = PosixError(filename, errno);
      }
    }
    ::close(fd);
    if (!status.ok()) {
      mmap_limiter_.Release();
    }
    return status;
  }

  Status NewWritableFile(const std::string& filename,
                         WritableFile** result) override {
    std::string path = "/dev/sda";

    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    // Open ZNS Device
    ssize_t ret = zbc_open(path.c_str(), O_RDWR, &dev);
    if (ret != 0) {
      if (ret == - ENODEV) 
        fprintf(stderr, "Open %s failed (not a zoned block device)\n", path.c_str());
      else
        fprintf(stderr, "Open %s failed (%s)\n", path.c_str(), strerror(-ret));
      return PosixError("NewWritableFile: zbc_open", errno);
    }

    // 먼저, 해당 filename 매핑된 zone 있는지 확인
    // nullptr: 없으면, filename 과 zone 매핑
    // 있다면 그대로 해당 target_zone 사용하면 됨.
    target_zone = leveldb::map_table[filename];
    if (target_zone == nullptr) {
      ret = FindZoneForFilename(dev, &target_zone);
      if (ret != 0) {
        fprintf(stderr, "NewWritableFile: FindZoneForFilename failed\n");
        return PosixError("NewWritableFile: FindZoneForFilename", errno);
      }

      // target_zone 함수 통해서 받아 왔을 때, filename 과 매핑.
      leveldb::map_table[filename] = target_zone;
    }

    *result = new PosixWritableFile(dev, target_zone, filename);

    return Status::OK();
  }

  Status NewAppendableFile(const std::string& filename,
                           WritableFile** result) override {
    std::string path = "/dev/sda";
    
    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    // Open ZNS Device
    ssize_t ret = zbc_open(path.c_str(), O_RDWR, &dev);
    if (ret != 0) {
      if (ret == - ENODEV) 
        fprintf(stderr, "Open %s failed (not a zoned block device)\n", path.c_str());
      else
        fprintf(stderr, "Open %s failed (%s)\n", path.c_str(), strerror(-ret));
      return PosixError(path, errno);
    }

    // 먼저, 해당 filename 매핑된 zone 있는지 확인
    // nullptr: 없으면, filename 과 zone 매핑
    // 있다면 그대로 해당 target_zone 사용하면 됨.
    target_zone = leveldb::map_table[filename];
    if (target_zone == nullptr) {
      ret = FindZoneForFilename(dev, &target_zone);
      if (ret != 0) {
        fprintf(stderr, "NewAppendableFile: FindZoneForFilename failed\n");
        return PosixError("NewAppendableFile: FindZoneForFilename", errno);
      }

      // target_zone 함수 통해서 받아 왔을 때, filename 과 매핑.
      leveldb::map_table[filename] = target_zone;
    }

    *result = new PosixWritableFile(dev, target_zone, filename);

    return Status::OK();
  }

  bool FileExists(const std::string& filename) override {
    // map 에서 key 로 가지고 있는 filename 이라면 true return
    // 그것이 아니라면 false
    return leveldb::map_table.find(filename) == leveldb::map_table.end();
  }

  Status GetChildren(const std::string& directory_path,
                     std::vector<std::string>* result) override {
    result->clear();
    // ::DIR* dir = ::opendir(directory_path.c_str());
    // if (dir == nullptr) {
    //   return PosixError(directory_path, errno);
    // }
    // struct ::dirent* entry;
    // while ((entry = ::readdir(dir)) != nullptr) {
    //   result->emplace_back(entry->d_name);
    // }
    // ::closedir(dir);
    return Status::OK();
  }

  Status RemoveFile(const std::string& filename) override {
    std::string path = "/dev/sda";

    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    // device open
    ssize_t ret = zbc_open(path.c_str(), O_RDWR, &dev);
    if (ret != 0) {
      if (ret == - ENODEV) 
        fprintf(stderr, "Open %s failed (not a zoned block device)\n", path.c_str());
      else
        fprintf(stderr, "Open %s failed (%s)\n", path.c_str(), strerror(-ret));
      return PosixError(path, errno);
    }

    // filename 과 매핑되어 있는 zone 가져 옴.
    target_zone = leveldb::map_table[filename];

    // target zone reset.
    ret = zbc_reset_zone(dev, target_zone->zbz_start, 0);
    if (ret == -EIO) {
      fprintf(stderr, "RemoveFile: zbc_reset_zone failed\n");
      zbc_close(dev);
      return PosixError("RemoveFile: zbc_reset_zone", errno);
    }

    return Status::OK();
  }

  Status CreateDir(const std::string& dirname) override {
    // if (::mkdir(dirname.c_str(), 0755) != 0) {
    //   return PosixError(dirname, errno);
    // }
    return Status::OK();
  }

  Status RemoveDir(const std::string& dirname) override {
    // if (::rmdir(dirname.c_str()) != 0) {
    //   return PosixError(dirname, errno);
    // }
    return Status::OK();
  }

  Status GetFileSize(const std::string& filename, uint64_t* size) override {
    std::string path = "/dev/sda";

    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    unsigned long long zone_ofst;
    unsigned long long zone_start;

    // filename 과 매핑 된 zone 의 현재 wp 받아오면
    // file 이 들어있는 block 크기 받을 수 있음.
    target_zone = leveldb::map_table[filename];
    if (target_zone == nullptr) {
      // target_zone 없다면, 일단 모든 zone reset 후 Error
      zbc_reset_zone(dev, 0, ZBC_OP_ALL_ZONES);
      zbc_close(dev);
      *size = 0;
      return PosixError("GetFileSize: no target zone for " + filename, errno);
    } 

    // 먼저, 해당 target_zone 의 현재 wp 와, start 구함.
    // 그 후, size 를 wp - start 로, 512B 단위로 얼마나 쓰여졌는지 구함.
    zone_ofst = zbc_zone_wp(target_zone);
    zone_start = zbc_zone_start(target_zone);
    
    *size = zone_ofst - zone_start;

    return Status::OK();
  }

  Status RenameFile(const std::string& from, const std::string& to) override {
    printf("RenameFile: \"%s\" to \"%s\" \n", from, to);

    struct zbc_zone *tmp_target_zone = map_table[from];
    map_table[to] = tmp_target_zone;

    // to 를 key 로 사용하여 교체 해주었으면
    // map_table 에서 from 제거
    map_table.erase(from);

    return Status::OK();
  }

  Status LockFile(const std::string& filename, FileLock** lock) override {
    *lock = nullptr;

    std::string path = "/dev/sda";
    struct zbc_device *dev = nullptr;
    struct zbc_zone *target_zone = nullptr;

    // Open ZBC Device
    ssize_t ret = zbc_open(path.c_str(), O_RDWR, &dev);
    if (ret != 0) {
      if (ret == - ENODEV) 
        fprintf(stderr, "Open %s failed (not a zoned block device)\n", path.c_str());
      else
        fprintf(stderr, "Open %s failed (%s)\n", path.c_str(), strerror(-ret));
      return PosixError(path, errno);
    }

    // 먼저, 해당 filename 매핑된 zone 있는지 확인
    // nullptr: 없으면, filename 과 zone 매핑
    // 있다면 그대로 해당 target_zone 사용하면 됨.
    target_zone = leveldb::map_table[filename];
    if (target_zone == nullptr) {
      ret = FindZoneForFilename(dev, &target_zone);
      if (ret != 0) {
        fprintf(stderr, "NewAppendableFile: FindZoneForFilename failed\n");
        return PosixError("NewAppendableFile: FindZoneForFilename", errno);
      }

      // target_zone 함수 통해서 받아 왔을 때, filename 과 매핑.
      leveldb::map_table[filename] = target_zone;
    }

    if (!locks_.Insert(filename)) {
      // ZBC Device Close.
      zbc_close(dev);
      return Status::IOError("lock " + filename, "already held by process");
    }

    if (LockOrUnlock(ret, true) == -1) {
      // ZBC Device Close.
      int lock_errno = errno;
      zbc_close(dev);
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
    // ZBC Device Close.
    zbc_close(posix_file_lock->dev());
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
    // CreateDir(*result);

    return Status::OK();
  }

  Status NewLogger(const std::string& filename, Logger** result) override {
    int fd = ::open(filename.c_str(),
                    O_APPEND | O_WRONLY | O_CREAT | kOpenBaseFlags, 0644);
    if (fd < 0) {
      *result = nullptr;
      return PosixError(filename, errno);
    }

    std::FILE* fp = ::fdopen(fd, "w");
    if (fp == nullptr) {
      ::close(fd);
      *result = nullptr;
      return PosixError(filename, errno);
    } else {
      *result = new PosixLogger(fp);
      return Status::OK();
    }
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
