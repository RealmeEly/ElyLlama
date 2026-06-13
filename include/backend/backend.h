#ifndef BACKEND_H
#define BACKEND_H

#if defined(_WIN32)
#include <windows.h>
#elif  defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include "utils/type.hpp"

#include <iostream>
#include <memory>
#include <vector>

class BackendInfo {
public:
  size_t id = 0;
  size_t total_memory = 0;
  size_t used_memory = 0;
  double compute_power = 0; // Relative computing power (such as TFLOPS) is used for load balancing
  double bandwidth = 0;     // PCIe/NVLink bandwidth (GB/s), used for copy estimation
  Device device = Device::CPU;

  [[nodiscard]] size_t availableMemory() const { return total_memory - used_memory; }
};

// ========== Backend Provider Interface ==========
class BackendProvider {
public:
  virtual ~BackendProvider() = default;

  [[nodiscard]] virtual bool isAvailable() const = 0;
  [[nodiscard]] virtual int getDeviceCount() const = 0;
  [[nodiscard]] virtual Device getBackendType() const = 0;
  [[nodiscard]] virtual const char* getBackendName() const = 0;
  [[nodiscard]] virtual BackendInfo getBackendInfo(int device_id) const = 0;
};

// ========== Backend Registry System ==========
class BackendRegistry {
private:
  BackendRegistry() = default;
  std::vector<std::unique_ptr<BackendProvider>> providers;

public:
  BackendRegistry(const BackendRegistry&) = delete;
  BackendRegistry& operator=(const BackendRegistry&) = delete;

  static BackendRegistry& instance() {
    static BackendRegistry registry;
    return registry;
  }

  void registerProvider(std::unique_ptr<BackendProvider> provider) {
    if (provider && provider->isAvailable()) {
      this->providers.push_back(std::move(provider));
    }
  }

  [[nodiscard]] const std::vector<std::unique_ptr<BackendProvider>>& getProviders() const {
    return this->providers;
  }
};

#if defined(BACKEND_CPU)
// ========== CPU Backend ==========
class CpuBackendProvider final : public BackendProvider {
public:
  [[nodiscard]] bool isAvailable() const override { return true; }
  [[nodiscard]] int getDeviceCount() const override { return 1; }
  [[nodiscard]] Device getBackendType() const override { return Device::CPU; }
  [[nodiscard]] const char* getBackendName() const override { return "CPU"; }
  [[nodiscard]] BackendInfo getBackendInfo(int device_id) const override;
  [[nodiscard]] static size_t getSystemMemory();
  [[nodiscard]] static size_t getSystemAvailableMemory();
  [[nodiscard]] static std::string getCpuName();
  [[nodiscard]] static uint32_t getPhysicalCores();
  [[nodiscard]] static uint32_t getThreadCount();
};
#endif

class DeviceManager {
private:
  mutable bool initialized = false;
  mutable std::vector<BackendInfo> devices;

  DeviceManager() = default;

  void detectDevices() const {
    devices.clear();
    for (const auto& registry = BackendRegistry::instance(); const auto& provider : registry.getProviders()) {
      const int count = provider->getDeviceCount();
      for (int i = 0; i < count; ++i) {
        devices.emplace_back(provider->getBackendInfo(i));
      }
    }
    initialized = true;
  }

public:
  static DeviceManager& instance() {
    static DeviceManager manager;
    return manager;
  }

  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;

  static void printDevices() {
    std::println(std::cout, "{:=<14} Detected Devices {:=<14}");
    for (const auto& registry = BackendRegistry::instance(); const auto& provider : registry.getProviders()) {
      for (int i = 0; i < provider->getDeviceCount(); ++i) {
        auto info = provider->getBackendInfo(i);
        std::println(std::cout, "[{}] {}  Memory: {:.1f} GB", info.id, provider->getBackendName(),
                     static_cast<double>(info.total_memory) / (1ULL << 30));
      }
      switch (provider->getBackendType()) {
#if defined(BACKEND_CPU)
        case Device::CPU: {
          auto info = provider->getBackendInfo(0);
          std::println(std::cout, "");
          std::println(std::cout, "  Device Name:       {}", CpuBackendProvider::getCpuName());
          std::println(std::cout, "  Physical Cores:    {}", CpuBackendProvider::getPhysicalCores());
          std::println(std::cout, "  Logical Threads:   {}", CpuBackendProvider::getThreadCount());
          std::println(std::cout, "  System Memory:     {:.1f} GB",
                       static_cast<double>(info.total_memory) / (1ULL << 30));
          std::println(std::cout, "  Available Memory:  {:.1f} GB",
                       static_cast<double>(info.availableMemory()) / (1ULL << 30));
          break;
        }
#endif
        default:
          std::println(std::cout, "");
      }
    }
  }

  [[nodiscard]] const std::vector<BackendInfo>& getDevices() const {
    if (!initialized) {
      detectDevices();
    }
    return devices;
  }

  [[nodiscard]] size_t deviceCount() const {
    return getDevices().size();
  }

  [[nodiscard]] const BackendInfo* getDevice(Device dev, size_t device_id) const {
    for (const auto& backend_info : getDevices()) {
      if (backend_info.device == dev && backend_info.id == device_id) {
        return &backend_info;
      }
    }
    return nullptr;
  }
};

#endif // BACKEND_H
