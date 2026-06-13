#ifndef RESOURCE_H
#define RESOURCE_H
#include "utils/type.hpp"

struct MemoryBlock {
  void* data_ptr = nullptr;
  size_t size = 0;
  size_t offset = 0;
  size_t device_handle = 0; // CPU: 0; Other: buffer handle
};

class MemoryResourceInterface { // NOLINT(*-special-member-functions)
public:
  virtual ~MemoryResourceInterface() = default;

  virtual void* allocate(size_t size, size_t alignment) = 0;
  virtual void deallocate(void* data_ptr, size_t size) = 0;
  [[nodiscard]] virtual size_t getId() const = 0;
  [[nodiscard]] virtual Device getDevice() const = 0;
  [[nodiscard]] virtual size_t getDeviceHandle() const { return 0; }
};

#endif // RESOURCE_H
