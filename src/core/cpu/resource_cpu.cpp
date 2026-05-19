// ReSharper disable CppDFAConstantConditions
// ReSharper disable CppDFAUnreachableCode
#include "core/cpu/resource_cpu.h"

#include <cstring>
#include <format>
#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cstdlib>
#include <sys/mman.h>
#endif

namespace core {

  void* CpuMemoryResource::allocate(size_t size, size_t alignment) {
    if (size == 0) {
      return nullptr;
    }
    void* data_ptr = nullptr;

#if defined(_WIN32)
    data_ptr = _aligned_malloc(size, alignment);
    if (data_ptr == nullptr) {
      throw std::runtime_error(std::format("CpuMemoryResource: _aligned_malloc {} bytes failed", size));
    }
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
      throw std::runtime_error(std::format("CpuMemoryResource: posix_memalign {} bytes (align={}) failed", size, alignment));
    }
#endif

    if (lock_memory) {
#if defined(_WIN32)
      if (VirtualLock(data_ptr, size) == 0) {
        std::cout << std::format("[warn] VirtualLock {} bytes failed, continuing without lock\n", size);
      }
#else
      if (mlock(ptr, size) != 0) {
        std::cout<<std::format("[warn] mlock {} bytes failed (try: ulimit -l unlimited), continuing without lock\n", size);
      }
#endif
    }

    std::memset(data_ptr, 0, size);
    return data_ptr;
  }

  void CpuMemoryResource::deallocate(void* data_ptr, size_t size) {
    if (data_ptr == nullptr) {
      return;
    }
    if (lock_memory && size > 0) {
#if defined(_WIN32)
      VirtualUnlock(data_ptr, size);
#else
      munlock(data_prt, size);
#endif
    }

#if defined(_WIN32)
    _aligned_free(data_ptr);
#else
    free(data_ptr);
#endif

    std::cout << std::format("Deallocated {} bytes on CPU\n", size);
  }
} // namespace core
