#ifndef GRAPH_H
#define GRAPH_H
#include "tensor.hpp"

#include <map>
#include <vector>

class ComputeGraph {
private:
  int max_layer = -1;
  std::vector<Tensor*> all_tensors;
  std::vector<Tensor*> execution_order;
  std::vector<Tensor*> external_outputs;
  std::map<int, std::vector<Tensor*>> layer_groups;
  // Group by dependency level, nodes in the same layer can be parallel
  std::vector<std::vector<Tensor*>> execution_levels;

  static std::string dotId(Tensor* t);
  void topologicalSort();
  void reserveBfsCollect(const std::vector<Tensor*>& seeds);

public:
  ComputeGraph() = default;

  ~ComputeGraph() {
    // ReSharper disable once CppLocalVariableMayBeConst
    for (Tensor* t : all_tensors) {
      delete t; // Only release metadata, actual data is managed by MemoryResourceInterface
    }
  }

  ComputeGraph(const ComputeGraph&) = delete;
  ComputeGraph& operator=(const ComputeGraph&) = delete;

  ComputeGraph(ComputeGraph&& other) noexcept;
  ComputeGraph& operator=(ComputeGraph&& other) noexcept;

  [[nodiscard]] const std::vector<Tensor*>& getExecutionOrder() const { return execution_order; }
  [[nodiscard]] const std::vector<std::vector<Tensor*>>& getExecutionLevels() const { return execution_levels; }
  [[nodiscard]] const std::vector<Tensor*>& getAllTensors() const { return all_tensors; }
  [[nodiscard]] const std::vector<Tensor*>& getExternalOutputs() const { return external_outputs; }
  [[nodiscard]] const std::map<int, std::vector<Tensor*>>& getLayerGroups() const { return layer_groups; }
  [[nodiscard]] int getMaxLayer() const { return max_layer; }

  void buildFromOutputs(std::initializer_list<Tensor*> outputs);
  void rebuildOrder();
  void clear();

  void replaceOutput(Tensor* old_t, Tensor* new_t);
  void addTensor(Tensor* t);
  void exportDot(const std::string& path) const;
  Tensor* insertMemcpy(Tensor* original_t, Device dst_device);
};

#endif // GRAPH_H
