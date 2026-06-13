#ifndef MODEL_H
#define MODEL_H
#include "core/gguf_parser.h"
#include "core/graph.h"
#include "utils/type.hpp"

#include <memory>

class ModelBase {
protected:
  ModelType type = ModelType::UNKNOWN;
  ModelArchType arch = ModelArchType::UNKNOWN;

public:
  std::string name = "unknown";
  virtual ~ModelBase() = default;
  virtual void printInfo() =0;
  [[nodiscard]] virtual size_t vocabSize() const =0;
  [[nodiscard]] virtual int64_t maxSeqLen() const =0;
  [[nodiscard]] virtual std::unique_ptr<ComputeGraph> buildGraph(const GGUFInfo&) =0;
};

class ModelFactory {
private:
  // register model create function
  using ModelCreator = std::unique_ptr<ModelBase>(*)();
  static inline std::unordered_map<std::string, ModelCreator> str_registry; // e.g. qwen3 -> Qwen3Model

  // register model helper function
  template <class ModelType>
  static std::unique_ptr<ModelBase> createModel() {
    return std::make_unique<ModelType>();
  }

public:
  // register model type
  template <class ModelType>
  static void registerModel(const std::string& arch_name) {
    // e.g. str_registry[qwen3] = std::make_unique<Qwen3Model>()
    str_registry[arch_name] = &createModel<ModelType>();
  }

  // create model from architecture
  [[nodiscard]] static std::unique_ptr<ModelBase> createFromArch(const std::string& arch_name) {
    const auto it = str_registry.find(arch_name);
    if (it == str_registry.end()) {
      throw std::runtime_error(std::format("Unsupported model architecture: {}", arch_name));
    }
    return it->second();
  }

  // create model from GGUFInfo
  [[nodiscard]] static std::unique_ptr<ModelBase> createFromGGUF(const GGUFInfo& info) {
    auto arch = info.getModelArchitecture();
    std::ranges::transform(arch, arch.begin(), [](const char c) { return std::tolower(c); });
    return createFromArch(arch);
  }

  // get registered models list
  [[nodiscard]] static std::vector<std::string> registeredModels() {
    std::vector<std::string> models;
    for (const auto& name : str_registry | std::views::keys) {
      models.push_back(name);
    }
    return models;
  }

  static void init();
};

#endif //MODEL_H
