#ifndef TENSOR_H
#define TENSOR_H

#include "utils/type.hpp"

#include <array>
#include <cstdint>

namespace core {
  class Tensor;

  constexpr int32_t TENSOR_MAX_SRC = 5;
  constexpr int32_t TENSOR_MAX_DIMS = 4;
  constexpr int32_t TENSOR_MAX_OP_PARAMS = 4;

  using Tensor_t = Tensor*;

  class Tensor {
    // NOLINTBEGIN(*-non-private-member-variables-in-classes)
  public:
    // ========== Cache line 0 (64B): memory position + graph topology ==========
    void* data = nullptr;     // CPU: really data pointer; Other: nullptr (use device_handle)
    size_t device_handle = 0; // CPU: 0; Other: buffer handle
    size_t offset = 0;        // data offset in memory pool (start from base)

    std::array<Tensor_t, TENSOR_MAX_SRC> src = {nullptr}; // source tensors

    // ========== Cache line 1: shape + dispatch info ==========
    std::array<int64_t, TENSOR_MAX_DIMS> dims = {};      // tensor shape
    OperationType op_type = OperationType::OP_TYPE_NONE; // op type
    Device device = Device::CPU;                         // devcice type, default is cpu
    DataType dtype = DataType::GGML_TYPE_F32;            // data type, default is f32
    TensorType type = TensorType::TENSOR_TYPE_UNKNOWN;   // tensor type
    int layer_id = -1;                                   // compute graph layer id

    // ========== Cache line 2: execute info ==========
    std::array<float, TENSOR_MAX_OP_PARAMS> op_params = {}; // op params
    std::array<uint64_t, TENSOR_MAX_DIMS> strides = {};     // op strides

    // ========== Cache line 3: cold data ==========
    std::string name; // tensor name

    // NOLINTEND(*-non-private-member-variables-in-classes)

    // ========== Functions ==========
    // Calculate number of elements
    [[nodiscard]] size_t numel() const {
      size_t total = 1;
      for (int i = 0; i < TENSOR_MAX_DIMS && dims[i] != 0; ++i) {
        total *= std::abs(dims[i]);
      }
      return total;
    }

    // Calculate static nbytes
    [[nodiscard]] size_t nbytes() const { return numel() * dataTypeSize(dtype); }

    // Calculate dynamic nbytes
    [[nodiscard]] size_t nbytesAt(const int64_t resolve) const {
      size_t total = 1;
      for (int i = 0; i < TENSOR_MAX_DIMS && dims[i] != 0; ++i) {
        total *= dims[i] < 0 ? resolve : dims[i];
      }
      return total * dataTypeSize(dtype);
    }

    // Return whether tensor is an operator
    [[nodiscard]] bool isComputed() const {
      return op_type != OperationType::OP_TYPE_NONE && type != TensorType::TENSOR_TYPE_WEIGHT && type !=
             TensorType::TENSOR_TYPE_INPUT;
    }
  };
} // namespace core

#endif // TENSOR_H
