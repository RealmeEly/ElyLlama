#include "core/graph.h"

#include <format>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

std::string ComputeGraph::dotId(Tensor* t) {
  return std::format("L{}_{}", t->layer_id, t->name);
}

void ComputeGraph::topologicalSort() {
  std::unordered_map<Tensor*, int> in_degree;
  std::unordered_multimap<Tensor*, Tensor*> forward;

  for (auto* t : all_tensors) {
    in_degree[t] = 0;
  }
  for (auto* t : all_tensors) {
    for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
      if (auto s = t->src[i]; s && in_degree.contains(s)) {
        in_degree[t]++;
        forward.emplace(s, t);
      }
    }
  }

  /* Grouping by layer: Nodes within the same level in_degree reset to zero,
   * run in parallel if independent of each other */
  execution_levels.clear();
  execution_order.clear();
  layer_groups.clear();
  max_layer = -1;
  size_t count = 0;

  while (count < all_tensors.size()) {
    // collect all node which in_degree==0 as same layer
    std::vector<Tensor*> level;
    for (auto* t : all_tensors) {
      if (in_degree[t] == 0) {
        level.push_back(t);
      }
    }

    if (level.empty()) { // graph has ring
      break;
    }

    // sort by layer_id within layer to maintain determinism
    std::sort(level.begin(), level.end(), [](const Tensor* a, const Tensor* b) {
      return a->layer_id < b->layer_id;
    });

    // remove nodes in this level to upgrade dependence
    for (auto* t : level) {
      in_degree[t] = -1; // mark as processed
      if (t->isOperator()) {
        execution_order.push_back(t);
        layer_groups[t->layer_id].push_back(t);
      }
      if (t->layer_id > max_layer) {
        max_layer = t->layer_id;
      }
      count++;
      auto [fst, snd] = forward.equal_range(t);
      for (auto it = fst; it != snd; ++it) {
        if (in_degree[it->second] >= 0) {
          in_degree[it->second]--;
        }
      }
    }
    execution_levels.push_back(std::move(level));
  }

  if (count != all_tensors.size()) {
    throw std::runtime_error(std::format("ComputeGraph: cycle detected, {}/{} resolved", count, all_tensors.size()));
  }
}

void ComputeGraph::reserveBfsCollect(const std::vector<Tensor*>& seeds) {
  std::pmr::unordered_set<Tensor*> visited;
  std::queue<Tensor*> q;

  for (auto* t : seeds) {
    if (t && visited.insert(t).second) {
      q.push(t);
    }
  }

  while (!q.empty()) {
    auto cur = q.front();
    q.pop();
    all_tensors.push_back(cur);
    for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
      if (!cur->src[i] && visited.insert(cur->src[i]).second) {
        q.push(cur->src[i]);
      }
    }
  }
}

ComputeGraph::ComputeGraph(ComputeGraph&& other) noexcept:
  max_layer(other.max_layer),
  all_tensors(std::move(other.all_tensors)),
  execution_order(std::move(other.execution_order)),
  external_outputs(std::move(other.external_outputs)),
  layer_groups(std::move(other.layer_groups)),
  execution_levels(std::move(other.execution_levels)) {
  other.max_layer = -1;
}

ComputeGraph& ComputeGraph::operator=(ComputeGraph&& other) noexcept {
  if (this != &other) {
    for (const auto* t : all_tensors) {
      delete t;
    }
    max_layer = other.max_layer;
    all_tensors = std::move(other.all_tensors);
    execution_order = std::move(other.execution_order);
    external_outputs = std::move(other.external_outputs);
    layer_groups = std::move(other.layer_groups);
    execution_levels = std::move(other.execution_levels);
    other.max_layer = -1;
  }
  return *this;
}

void ComputeGraph::buildFromOutputs(const std::initializer_list<Tensor*> outputs) {
  clear();
  external_outputs.assign(outputs);
  reserveBfsCollect(external_outputs);
  topologicalSort();
  // set node device to unknown before assign device
  for (auto* t : all_tensors) {
    t->device = Device::UNKNOWN;
  }
}

void ComputeGraph::rebuildOrder() {
  execution_order.clear();
  execution_levels.clear();
  layer_groups.clear();
  max_layer = -1;
  topologicalSort();
}

void ComputeGraph::clear() {
  for (const auto* t : all_tensors) {
    delete t;
  }
  all_tensors.clear();
  execution_order.clear();
  execution_levels.clear();
  external_outputs.clear();
  layer_groups.clear();
  max_layer = -1;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
void ComputeGraph::replaceOutput(Tensor* old_t, Tensor* new_t) {
  for (auto& out : external_outputs) {
    if (out == old_t) {
      out = new_t;
    }
  }
}

void ComputeGraph::addTensor(Tensor* t) {
  all_tensors.push_back(t);
}

// export compute graph to visibility graph
void ComputeGraph::exportDot(const std::string& path) const {
  std::ofstream os(path);
  if (!os)
    throw std::runtime_error(std::format("Cannot open file: {}", path));

  os << "digraph ComputeGraph {\n"
      << "  rankdir=TB;\n"
      << "  node [shape=box, style=filled];\n\n";

  for (auto* t : all_tensors) {
    const char* color = [t] {
      if (t->type == TensorType::TENSOR_TYPE_WEIGHT)
        return "#FFD54F";
      if (t->type == TensorType::TENSOR_TYPE_INPUT)
        return "#90CAF9";
      if (t->type == TensorType::TENSOR_TYPE_OUTPUT)
        return "#ff0000";
      if (t->type == TensorType::TENSOR_TYPE_CACHE)
        return "#4a91d8";
      return "#C8E6C9";
    }();

    std::string layer_prefix = (t->layer_id >= 0) ?
                                 std::format("[L{},{}] ", t->layer_id, dataTypeToString(t->dtype)) :
                                 std::format("[Global,{}] ", dataTypeToString(t->dtype));
    std::string label = layer_prefix + t->name;
    std::vector<int64_t> real_dims;
    for (int64_t dim : t->dims) {
      if (dim != 0)
        real_dims.push_back(dim);
    }
    label += std::format("\\n{}{}", deviceToString(t->device), real_dims);
    if (t->op_type != OperationType::OP_TYPE_NONE)
      label += "\\n" + operationTypeToString(t->op_type);
    os << std::format("  \"{}\" [fillcolor=\"{}\", label=\"{}\"];\n", dotId(t), color, label);
  }
  os << "\n";
  for (auto* t : all_tensors) {
    for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
      if (!t->src[i])
        continue;
      const char* style = (t->type == TensorType::TENSOR_TYPE_VIEW) ? "style=dashed" : "";
      os << std::format("  \"{}\" -> \"{}\" [{}];\n", dotId(t->src[i]), dotId(t), style);
    }
  }

  os << "}\n";
}

Tensor* ComputeGraph::insertMemcpy(Tensor* original_t, Device dst_device) {
  const auto proxy = new Tensor();
  proxy->name = "memcpy_" + original_t->name;
  proxy->op_type = OperationType::OP_TYPE_MEMCPY;
  proxy->type = original_t->type;
  proxy->device = dst_device;
  proxy->dtype = original_t->dtype;
  proxy->dims = original_t->dims;
  proxy->layer_id = original_t->layer_id;
  proxy->src[0] = original_t;
  addTensor(proxy);
  return proxy;
}
