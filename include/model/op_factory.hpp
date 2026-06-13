#ifndef OP_FACTORY_HPP
#define OP_FACTORY_HPP

#include "core/tensor.hpp"
#include "utils/tools.hpp"

class OpFactory {
  using Tensor = Tensor;

  static void computeStrides(Tensor* t) {
    size_t stride = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
      if (t->dims[i] == 0) {
        t->strides[i] = 0;
      } else {
        t->strides[i] = stride * dataTypeSize(t->dtype);
        stride *= t->dims[i];
      }
    }
  }

public:
  static Tensor* placeHolder(DataType dtype, TensorType type, std::initializer_list<int64_t> dims,
                             const std::string& name, int layer_id = -1) {
    if (dims.size() > 4) {
      logError("dims size must be 1~4");
      throw std::runtime_error("dims size must be 1~4");
    }

    auto* t = new Tensor(name, layer_id, dtype, type, OperationType::OP_TYPE_NONE);
    std::ranges::copy(dims, t->dims.begin());
    computeStrides(t);
    t->data = nullptr;
    return t;
  }
};

#endif //OP_FACTORY_HPP
