#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>

typedef enum LogLevel {
  INFO  = 0,
  WARN  = 1,
  ERROR = 2
} LogLevel;

// ANSI color code
constexpr std::string COLOR_RESET = "\033[0m";
constexpr std::string COLOR_RED = "\033[31m";
constexpr std::string COLOR_YELLOW = "\033[33m";
constexpr std::string COLOR_CYAN = "\033[36m";

inline void localTime(std::time_t& t, std::tm& tm_info) {
#ifdef _WIN32
  localtime_s(&tm_info, &t);
#else
  localtime_r(&t, &tm_info);
#endif
}

inline void logBase(LogLevel level, const std::string& color, const std::string& msg) {
  const auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm_info{};
  localTime(t, tm_info);
  std::cerr << color << "[" << level << "] "
      << std::put_time(&tm_info, "%Y-%m-%d %H:%M:%S")
      << " [" << __FILE__ << ":" << __LINE__ << "]"
      << " (" << __func__ << "): " << msg << COLOR_RESET << std::endl;
}

inline void logInfo(const std::string& msg) {
  logBase(INFO, COLOR_CYAN, msg);
}

inline void logWarn(const std::string& msg) {
  logBase(WARN, COLOR_YELLOW, msg);
}

inline void logError(const std::string& msg) {
  logBase(ERROR, COLOR_RED, msg);
}

inline void printTime(const double ms) {
  std::cout << COLOR_RED << std::fixed << std::setprecision(3);
  if (ms < 1000.0) {
    std::cout << "Execution time: " << ms << "ms";
  } else if (ms < 60.0 * 1000.0) {
    std::cout << "Execution time: " << ms / 1000.0 << "s";
  } else if (ms < 60.0 * 60.0 * 1000.0) {
    std::cout << "Execution time: " << ms / (60.0 * 1000.0) << "min";
  } else {
    std::cout << "Execution time: " << ms / (60.0 * 60.0 * 1000.0) << "h";
  }
  std::cout << COLOR_RESET << std::endl;
}

template <typename F>
auto runningTime(F&& f) {
  const auto start = std::chrono::steady_clock::now();
  using Result = decltype(f());
  if constexpr (std::is_void_v<Result>) {
    f();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> duration_ms = end - start;
    const double ms = duration_ms.count();
    printTime(ms);
  } else {
    auto result = f();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> duration_ms = end - start;
    const double ms = duration_ms.count();
    printTime(ms);
    return result;
  }
}
#endif //TOOLS_HPP
