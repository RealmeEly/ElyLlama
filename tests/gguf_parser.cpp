#include "core/gguf_parser.h"

#include <cxxopts.hpp>
#include <iostream>

// ReSharper disable once CppDFAConstantFunctionResult, CppParameterMayBeConst
int main(int argc, char* argv[]) {
  cxxopts::Options options("test_gguf_parser", "test gguf file parser");
  options.add_options()("m,model", "Model file path", cxxopts::value<std::string>())("h,help", "Print usage");

  const auto result = options.parse(argc, argv);
  if (result.contains("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (!result.contains("model")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  {
    const auto file_path = result["model"].as<std::string>();
    core::GGUFParser parser(file_path);
    const auto info = parser.getInfo();
    info.printInfo();
    std::println(std::cout, "Destructing...");
  }

  std::println(std::cout, "Destruct finished");

  return 0;
}
