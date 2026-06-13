#include "backend/backend.h"

// ReSharper disable CppUnusedIncludeDirective
#include <cstddef>
#include <fstream>
#include <unordered_set>
// ReSharper restore CppUnusedIncludeDirective

#if defined()_WIN32
#include <windows.h>
#endif

BackendInfo CpuBackendProvider::getBackendInfo(int device_id) const {
  const size_t total = getSystemMemory();
  const size_t available = getSystemAvailableMemory();
  const size_t used = total - available;
  return BackendInfo(0, total, used, 0, 0, Device::CPU);
}

size_t CpuBackendProvider::getSystemMemory() {
#if defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
#elif  defined(__linux__)|| defined(__APPLE__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
      return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    }
    return 8ULL << 30;
#else
    return 8ULL << 30;
#endif
}

size_t CpuBackendProvider::getSystemAvailableMemory() {
#if defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullAvailPhys;
#elif defined(__linux__)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
      return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    }
    return 8ULL << 30;
#elif defined(__APPLE__)
    int64_t available = 0;
    size_t len = sizeof(available);
    if (sysctlbyname("vm.page_free_count", nullptr, &len, nullptr, 0) == 0) {
      long page_size = sysconf(_SC_PAGE_SIZE);
      int64_t free_pages = 0;
      sysctlbyname("vm.page_free_count", &free_pages, &len, nullptr, 0);
      return static_cast<size_t>(free_pages) * static_cast<size_t>(page_size);
    }
    return 8ULL << 30;
#else
    return 8ULL << 30;
#endif
}

std::string CpuBackendProvider::getCpuName() {
#if defined(_WIN32)
  HKEY h_key;
  // ReSharper disable once CppLocalVariableMayBeConst
  char cpu_name[256] = "Unknown CPU";
  DWORD size = sizeof(cpu_name);
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, R"(HARDWARE\DESCRIPTION\System\CentralProcessor\0)", 0, KEY_READ, &h_key) ==
      ERROR_SUCCESS) {
    RegQueryValueExA(h_key, "ProcessorNameString", nullptr, nullptr, reinterpret_cast<LPBYTE>(cpu_name), &size);
    RegCloseKey(h_key);
  }
  return cpu_name;
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("model name") == 0) {
        size_t pos = line.find(':');
        if (pos != std::string::npos && pos + 2 < line.length()) {
          return line.substr(pos + 2);
        }
      }
    }
    return "Unknown CPU";
#elif defined(__APPLE__)
    char cpuName[256];
    size_t len = sizeof(cpuName);
    if (sysctlbyname("machdep.cpu.brand_string", cpuName, &len, NULL, 0) == 0) {
      return cpuName;
    }
    return "Unknown CPU";
#else
    return "Unknown CPU";
#endif
}

uint32_t CpuBackendProvider::getPhysicalCores() {
#if defined(_WIN32)
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  return system_info.dwNumberOfProcessors;
#elif defined(__linux__) || defined(__APPLE__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<unsigned int>(n) : 1;
#else
    return 1;
#endif
}

uint32_t CpuBackendProvider::getThreadCount() {
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  DWORD length = 0;
  GetLogicalProcessorInformation(nullptr, &length);
  if (length == 0)
    return si.dwNumberOfProcessors;
  const auto buffer = std::make_unique<char[]>(length);
  auto info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(buffer.get());
  if (!GetLogicalProcessorInformation(info, &length))
    return si.dwNumberOfProcessors;
  unsigned int count = 0;
  size_t offset = 0;
  while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= length) {
    if (info->Relationship == RelationProcessorCore)
      count++;
    offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    info++;
  }
  return count > 0 ? count : si.dwNumberOfProcessors;
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::unordered_set<std::string> cores;
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("core id") == 0) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
          cores.insert(line.substr(pos + 2));
        }
      }
    }
    return cores.empty() ? get_thread_count() : static_cast<unsigned int>(cores.size());
#elif defined(__APPLE__)
    return get_thread_count();
#else
    return 1;
#endif
}

static class CpuBackendProviderRegistrar {
public:
  CpuBackendProviderRegistrar() {
    BackendRegistry::instance().registerProvider(std::make_unique<CpuBackendProvider>());
  }
} global_cpu_backend_registrar;
