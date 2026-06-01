#ifndef GRAPH_H
#define GRAPH_H
#include "tensor.hpp"

#include <map>
#include <vector>

namespace core {
  class ComputeGraph {
  private:
    int max_layer = -1;
    std::vector<Tensor_t> all_tensors;
    std::vector<Tensor_t> execution_order;
    std::vector<Tensor_t> external_outputs;
    std::map<int, std::vector<Tensor_t>> layer_groups;
    // Group by dependency level, nodes in the same layer can be parallel
    std::vector<std::vector<Tensor_t>> execution_levels;
  };
} // namespace core

#endif // GRAPH_H
