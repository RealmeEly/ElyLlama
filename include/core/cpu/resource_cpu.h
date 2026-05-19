#ifndef RESOURCE_CPU_H
#define RESOURCE_CPU_H
#include "core/resource.h"

namespace core {
  class CpuMemoryResource final : public MemoryResourceInterface {
  private:
    bool lock_memory = false;

  public:
    // ReSharper disable once CppParameterMayBeConst
    explicit CpuMemoryResource(bool lock = false):
      lock_memory(lock) {
    }

    void* allocate(size_t size, size_t alignment) override;

    void deallocate(void* data_ptr, size_t size) override;

    [[nodiscard]] size_t getId() const override { return 0; }

    [[nodiscard]] Device getDevice() const override { return Device::CPU; }

    [[nodiscard]] bool getLockMemory() const { return lock_memory; }
  };
} // namespace core

#endif //RESOURCE_CPU_H
