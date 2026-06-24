#ifndef OP_FACTORY_HPP
#define OP_FACTORY_HPP

#include "core/gguf_parser.h"
#include "core/tensor.hpp"
#include "utils/tools.hpp"

class OpFactory {
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

  // infer linear output tensor shape
  static std::array<int64_t, TENSOR_MAX_DIMS> inferLinearOutputShape(std::span<const int64_t> input_dims,
                                                                     std::span<const int64_t> weight_dims,
                                                                     bool transpose) {
    std::array<int64_t, TENSOR_MAX_DIMS> output_dims{};
    std::ranges::fill(output_dims, 0);

    int input_ndim = 0;
    for (int i = static_cast<int>(input_dims.size()) - 1; i >= 0; --i) {
      if (input_dims[i] != 0) {
        input_ndim = i + 1;
        break;
      }
    }

    if (input_ndim == 0) {
      logError("Empty input shape");
      throw std::runtime_error("Empty input shape");
    }

    auto weight_in = weight_dims[0];
    auto weight_out = weight_dims[1];
    if (transpose) {
      std::swap(weight_in, weight_out);
    }

    if (auto input_last = input_dims[input_ndim - 1]; input_last > 0 && weight_in > 0 && input_last != weight_in) {
      logError(std::format("Linear dimension mismatch: input last={} vs weight in={}", input_last, weight_in));
      throw std::runtime_error(std::format("Linear dimension mismatch: input last={} vs weight in={}", input_last,
                                           weight_in));
    }

    for (int i = 0; i < input_ndim - 1; ++i) {
      output_dims[i] = input_dims[i];
    }
    output_dims[input_ndim - 1] = weight_out;
    return output_dims;
  }

public: // ReSharper disable CppDFAMemoryLeak
  static const TensorInfo* findTensor(const GGUFInfo& info, const std::string& name) {
    for (const auto& t : info.tensor_info_vec) {
      if (t.name == name) {
        return &t;
      }
    }
    logError(std::format("Tensor not found: {}", name));
    throw std::runtime_error(std::format("Tensor not found: {}", name));
  }

  static Tensor* placeHolder(DataType dtype, TensorType type, std::initializer_list<int64_t> dims,
                             const std::string& name, int layer_id = -1) {
    if (dims.size() > 4) {
      logError("dims size must be 1~4");
      throw std::runtime_error("dims size must be 1~4");
    }

    auto* t = new Tensor(name, layer_id, dtype, type, OperationType::OP_TYPE_NONE);
    std::ranges::copy(dims, t->dims.begin());
    computeStrides(t);
    return t;
  }

  static Tensor* weightPlaceHolder(const TensorInfo* info, const std::string& name, int layer_id = -1) {
    auto* t = new Tensor(name, layer_id, info->dtype, TensorType::TENSOR_TYPE_WEIGHT, OperationType::OP_TYPE_NONE);
    t->offset = info->offset;
    std::ranges::copy(info->dimensions, t->dims.begin());
    return t;
  }

  // linear: y = x @ weight + bias
  static Tensor* linear(Tensor* input, const TensorInfo* weight_info, const std::string& name = "", int layer_id = -1,
                        bool transpose = false, Tensor* bias = nullptr) {
    auto* t = new Tensor(name, layer_id, input->dtype, TensorType::TENSOR_TYPE_ACTIVATION,
                         OperationType::OP_TYPE_LINEAR);

    auto output_shape = inferLinearOutputShape({input->dims.begin(), input->dims.end()}, {weight_info->dimensions},
                                               transpose);
    std::ranges::copy(output_shape, t->dims.begin());

    t->src[0] = input;
    t->src[1] = weightPlaceHolder(weight_info, weight_info->name, layer_id);
    t->src[2] = bias;
    t->op_params[0] = transpose ? 1 : 0;
    computeStrides(t);
    return t;
  }

  static Tensor* embeddingLookup(Tensor* input_ids, const TensorInfo* weight_info, const std::string& name = "",
                                 int layer_id = -1, bool transpose = false) {
    auto* t = new Tensor(name, layer_id, weight_info->dtype, TensorType::TENSOR_TYPE_ACTIVATION,
                         OperationType::OP_TYPE_EMBEDDING);

    t->dims[0] = input_ids->dims[0];                                                  // batch
    t->dims[1] = input_ids->dims[1];                                                  // seq_len
    t->dims[2] = transpose ? weight_info->dimensions[0] : weight_info->dimensions[1]; // hidden_size

    t->src[0] = input_ids;
    t->src[1] = weightPlaceHolder(weight_info, weight_info->name, layer_id);
    t->op_params[0] = transpose ? 1 : 0;
    computeStrides(t);
    return t;
  }

  // rms_norm: y = x / sqrt[mean(x^2) + eps] * weight
  static Tensor* rmsNorm(Tensor* input, const TensorInfo* weight_info, const std::string& name = "", int layer_id,
                         float eps) {
    auto* t = new Tensor(name, layer_id, input->dtype, TensorType::TENSOR_TYPE_ACTIVATION,
                         OperationType::OP_TYPE_RMS_NORM);
    std::ranges::copy(input->dims, t->dims.begin());

    t->src[0] = input;
    t->src[1] = weightPlaceHolder(weight_info, weight_info->name, layer_id);
    t->op_params[0] = eps;
    computeStrides(t);
    return t;
  }

  static std::tuple<Tensor*, Tensor*> ropeCache(int max_seq_len, int head_dim, float theta, DataType dtype,
                                                const std::string& name_prefix = "rope", int layer_id = -1) {
    auto make_cache_tensor = [&](const std::string& name)-> Tensor* {
      auto* t = new Tensor(name, layer_id, dtype, TensorType::TENSOR_TYPE_CACHE, OperationType::OP_TYPE_ROPE_CACHE);
      std::ranges::fill(t->dims, 0);
      t->dims[0] = max_seq_len;
      t->dims[1] = head_dim;
      t->op_params[0] = theta;
      t->op_params[1] = static_cast<float>(head_dim);
      t->op_params[2] = static_cast<float>(max_seq_len);
      computeStrides(t);
      return t;
    };

    auto* cos_tensor = make_cache_tensor(name_prefix + "_cos");
    auto* sin_tensor = make_cache_tensor(name_prefix + "_sin");
    return {cos_tensor, sin_tensor};
  }

  // silu: y = x * sigmoid(x)
  static Tensor* silu(Tensor* input, const std::string& name = "", int layer_id = -1) {
    auto* t = new Tensor(name, layer_id, input->dtype, TensorType::TENSOR_TYPE_ACTIVATION, OperationType::OP_TYPE_SILU);
    std::ranges::copy(input->dims, t->dims.begin());
    t->src[0] = input;
    computeStrides(t);
    return t;
  }

  // sigmoid: y = sigmoid(x)
  static Tensor* sigmoid(Tensor* input, const std::string& name = "", int layer_id = -1) {
    auto* t = new Tensor(name, layer_id, input->dtype, TensorType::TENSOR_TYPE_ACTIVATION,
                         OperationType::OP_TYPE_SIGMOID);
    std::ranges::copy(input->dims, t->dims.begin());
    t->src[0] = input;
    computeStrides(t);
    return t;
  }

  // narrow: take a slice along one dimension (view, no copy), op_params: [0]=dim, [1]=start, [2]=size
  static Tensor* narrow(Tensor* input, const std::string& name = "", int layer_id = -1, int dim, int64_t start,
                        int64_t size) {
    auto* t = new Tensor(name, layer_id, input->dtype, TensorType::TENSOR_TYPE_ACTIVATION,
                         OperationType::OP_TYPE_NARROW);
    std::ranges::copy(input->dims, t->dims.begin());
    t->dims[dim] = size;

    t->src[0] = input;
    t->op_params[0] = static_cast<float>(dim);
    t->op_params[1] = static_cast<float>(start);
    t->op_params[2] = static_cast<float>(size);
    computeStrides(t);
    return t;
  }

  static Tensor* add(Tensor* a, Tensor* b, const std::string& name = "", int layer_id = -1) {
    auto* t = new Tensor(name, layer_id, a->dtype, TensorType::TENSOR_TYPE_ACTIVATION, OperationType::OP_TYPE_ADD);
    std::ranges::copy(a->dims, t->dims.begin());
    t->src[0] = a;
    t->src[1] = b;
    computeStrides(t);
    return t;
  }

  static Tensor* mul(Tensor* a, Tensor* b, const std::string& name = "", int layer_id = -1) {
    for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
      if (a->dims[i] != b->dims[i] && a->dims[i] > 0 && b->dims[i] > 0) {
        logError("mul: shape mismatch");
        throw std::runtime_error("mul: shape mismatch");
      }
    }

    auto* t = new Tensor(name.empty() ? a->name + "_mul_" + b->name : name, layer_id, a->dtype,
                         TensorType::TENSOR_TYPE_ACTIVATION, OperationType::OP_TYPE_MUL);
    std::ranges::copy(a->dims, t->dims.begin());
    t->src[0] = a;
    t->src[1] = b;
    computeStrides(t);
    return t;
  }

  static Tensor* reshape(Tensor* input, const std::string& name = "", int layer_id = -1,
                         std::initializer_list<int64_t> new_shape_init) {
    std::vector new_shape(new_shape_init);

    // collect source valid dimension
    std::vector<int64_t> src_shape;
    for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
      if (input->dims[i] != 0) {
        src_shape.push_back(input->dims[i]);
      }
    }

    // handle dynamic dimension
    int64_t src_nelem = 1;
    bool src_has_dynamic = false;
    for (const auto d : src_shape) {
      if (d == -1) {
        src_has_dynamic = true;
      } else {
        src_nelem *= d;
      }
    }

    int64_t known_nelem = 1;
    int infer_idx = -1;
    for (int i = 0; i < new_shape.size(); ++i) {
      if (new_shape[i] == -1) {
        infer_idx = i;
      } else {
        known_nelem *= new_shape[i];
      }
    }
    if (infer_idx >= 0) {
      if (src_has_dynamic) {
        new_shape[infer_idx] = -1; // keep dynamic
      } else {
        if (src_nelem % known_nelem != 0) {
          logError(std::format("Reshape invalid: {} elements cannot fit into shape", src_nelem));
          throw std::runtime_error(std::format("Reshape invalid: {} elements cannot fit into shape", src_nelem));
        }
        new_shape[infer_idx] = src_nelem / known_nelem;
      }
    }

    // verify (skip dynamic dimension)
    if (!src_has_dynamic) {
      int64_t target_nelem = 1;
      for (const auto d : new_shape) {
        if (d > 0) {
          target_nelem *= d;
        }
      }
      if (src_nelem != target_nelem) {
        logError(std::format("Reshape mismatch: src elements = {} and target elements = {}", src_nelem, target_nelem));
        throw std::runtime_error(std::format("Reshape mismatch: src elements = {} and target elements = {}", src_nelem,
                                             target_nelem));
      }
    }

    auto* view = new Tensor(name.empty() ? input->name + "_reshape" : name, layer_id, input->dtype,
                            TensorType::TENSOR_TYPE_VIEW, OperationType::OP_TYPE_RESHAPE);
    std::ranges::fill(view->dims, 0);
    std::ranges::copy(new_shape, view->dims.begin());
    view->data = input->data;
    view->offset = input->offset;
    view->src[0] = input;
    computeStrides(view);
    return view;
  }

  static Tensor* permute(Tensor* input, const std::string& name = "", int layer_id = -1,
                         std::initializer_list<int> perm_list) {
    if (!input) {
      return nullptr;
    }

    std::vector perm(perm_list);

    // collect source valid dimension
    std::vector<int64_t> src_shape;
    for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
      if (input->dims[i] != 0) {
        src_shape.push_back(input->dims[i]);
      }
    }
    const size_t src_ndim = src_shape.size();

    // verify perm range
    if (perm.size() != src_ndim) {
      logError(std::format("Permute dim mismatch: perm rank = {} and src rank = {}", perm.size(), src_ndim));
      throw std::runtime_error(std::format("Permute dim mismatch: perm rank = {} and src rank = {}", perm.size(),
                                           src_ndim));
    }
    for (int& i : perm) {
      if (i < 0 || i >= src_ndim) {
        logError(std::format("Permute index {} out of range [0, {})", i, src_ndim));
        throw std::runtime_error(std::format("Permute index {} out of range [0, {})", i, src_ndim));
      }
    }

    auto* t = new Tensor(name.empty() ? input->name + "_permute" : name, layer_id, input->dtype,
                         TensorType::TENSOR_TYPE_ACTIVATION, OperationType::OP_TYPE_PERMUTE);
    std::ranges::fill(t->dims, 0);
    std::ranges::fill(t->op_params, 0.0F);
    for (int i = 0; i < perm.size(); ++i) {
      t->dims[i] = src_shape[perm[i]];
      t->op_params[i] = static_cast<float>(perm[i]);
    }
    t->src[0] = input;
    computeStrides(t);
    return t;
  }

  static Tensor* reshapePermute(Tensor* input, const std::string& name = "", int layer_id = -1,
                                std::initializer_list<int64_t> new_shape_init, std::initializer_list<int> perm_list) {
    auto* reshaped = reshape(input, name + "_reshape", layer_id, new_shape_init);
    if (!reshaped) {
      return nullptr;
    }
    auto* result = permute(reshaped, name + "_permute", layer_id, perm_list);
    return result;
  }

  static Tensor* permuteReshape(Tensor* input, const std::string& name = "", int layer_id = -1,
                                std::initializer_list<int> perm_list,
                                std::initializer_list<int64_t> new_shape_init) {
    auto* permuted = permute(input, name + "_permute", layer_id, perm_list);
    if (!permuted) {
      return nullptr;
    }
    auto* result = reshape(permuted, name + "_reshape", layer_id, new_shape_init);
    return result;
  }

  //
  static Tensor* scaledDotProductAttention(
    Tensor* q_rope,
    Tensor* k_rope,
    Tensor* v_4d,
    Tensor* mask = nullptr,
    const std::string& name,
    int layer_id = -1,
    float scale = -1.0F,
    bool causal = true,
    int64_t num_kv_groups = 1 // GQA = n_heads / n_kv_heads
  ) {
    if (!q_rope || !k_rope || !v_4d) {
      logError("SDPA: null input tensor");
      throw std::runtime_error("SDPA: null input tensor");
    }

    // verify dimension
    int64_t n_heads = q_rope->dims[1];
    int64_t n_kv_heads = k_rope->dims[1];
    if (n_heads % n_kv_heads != 0) {
      logError(std::format("SDPA: n_heads = {} must be divisiable by n_kv_heads = {}", n_heads, n_kv_heads));
      throw std::runtime_error(std::format("SDPA: n_heads = {} must be divisiable by n_kv_heads = {}", n_heads,
                                           n_kv_heads));
    }

    if (num_kv_groups < 1) {
      num_kv_groups = n_heads / n_kv_heads;
    }

    auto* t = new Tensor(name.empty() ? "sdpa_out" : name, layer_id, q_rope->dtype, TensorType::TENSOR_TYPE_ACTIVATION,
                         OperationType::OP_TYPE_SDPA);

    // output shape as same as q_rope: [batch, n_heads, seq_len, head_dim]
    std::ranges::copy(q_rope->dims, t->dims.begin());

    // view: shared memory
    t->data = q_rope->data;
    t->offset = q_rope->offset;

    t->src[0] = q_rope;
    t->src[1] = k_rope;
    t->src[2] = v_4d;
    t->src[3] = mask;

    std::ranges::copy(q_rope->strides, t->strides.begin());

    const int64_t head_dim = q_rope->dims[3];
    const int32_t head_dim_i32 = static_cast<int32_t>(head_dim);
    const int32_t kv_groups_i32 = static_cast<int32_t>(num_kv_groups);
    const float scale_val = (scale < 0) ? (1.0F / std::sqrt(static_cast<float>(head_dim))) : scale;

    const int32_t causal_i32 = causal ? 1 : 0;

    t->op_params[0] = static_cast<float>(head_dim_i32);
    t->op_params[1] = scale_val;
    t->op_params[2] = static_cast<float>(causal_i32);
    t->op_params[3] = static_cast<float>(kv_groups_i32);

    return t;
  }

  static std::tuple<Tensor*, Tensor*> applyRope(
    Tensor* q,
    Tensor* k,
    Tensor* cos_cahce,
    Tensor* sin_cache,
    Tensor* position_ids = nullptr,
    const std::string& name_suffix = "",
    int layer_id = -1
  ) {
    if (!q || !k || !sin_cache || !cos_cahce) {
      logError("applyRope: nullptr input tensor");
      throw std::runtime_error("applyRope: nullptr input tensor");
    }

    int64_t head_dim = 0;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
      if (q->dims[i] > 0) {
        head_dim = q->dims[i];
        break;
      }
    }
    if (head_dim <= 0 || head_dim % 2 != 0) {
      logError("applyRope: invalid head_dim");
      throw std::runtime_error("applyRope: invalid head_dim");
    }

    int64_t rope_dim = cos_cahce->dims[1];
    if (rope_dim > head_dim || rope_dim % 2 != 0) {
      logError(std::format("applyRope: invalid cache dim, rope_dim = {}, head_dim = {}", rope_dim, head_dim));
      throw std::runtime_error(std::format("applyRope: invalid cache dim, rope_dim = {}, head_dim = {}", rope_dim,
                                           head_dim));
    }

    auto make_rope_output = [&](Tensor* input, const std::string& name)-> Tensor* {
      auto* t = new Tensor(name, layer_id, input->dtype, TensorType::TENSOR_TYPE_ACTIVATION,
                           OperationType::OP_TYPE_APPLY_ROPE);
      std::ranges::copy(input->dims, t->dims.begin());
      t->data = input->data;
      t->offset = input->offset;
      t->src[0] = input;
      t->src[1] = cos_cahce;
      t->src[2] = sin_cache;
      t->src[3] = position_ids;
      std::ranges::copy(input->strides, t->strides.begin());
      t->op_params[0] = static_cast<float>(head_dim);
      t->op_params[1] = static_cast<float>(rope_dim);
      return t;
    };

    const std::string suffix = name_suffix.empty() ? "" : "_" + name_suffix;
    auto* q_rope = make_rope_output(q, q->name + "_rope" + suffix);
    auto* k_rope = make_rope_output(k, k->name + "_rope" + suffix);
    return {q_rope, k_rope};
  }

  // causal one-dimensional convolution (input projection with Mamba2)
  static Tensor* causalConv1d(
    Tensor* input,                 // [batch, seq_len, dim_inner]
    const TensorInfo* weight_info, // [kernel, dim_inner]
    const std::string& name = "",
    int layer_id = -1,
    int kernel_size
  ) {
    auto* t = new Tensor(name, layer_id, input->dtype, TensorType::TENSOR_TYPE_ACTIVATION,
                         OperationType::OP_TYPE_CAUSAL_CONV1D);
    std::ranges::copy(input->dims, t->dims.begin());
    t->src[0] = input;
    t->src[1] = weightPlaceHolder(weight_info, weight_info->name, layer_id);
    t->op_params[0] = static_cast<float>(kernel_size);
    computeStrides(t);
    return t;
  }

  // Mamba2 selective scan / chunk scan abstract nodes
  static Tensor* ssmScan(
    Tensor* input,                  // [batch, seq_len, dim_inner] (conv_out)
    const TensorInfo* a_info,       // ssm_a
    const TensorInfo* alpha_info,   // ssm_alpha
    const TensorInfo* beta_info,    // ssm_beta
    const TensorInfo* dt_bias_info, // ssm_dt.bias
    const std::string& name = "",
    int layer_id = -1,
    int64_t output_inner_size = 0 // 0 means the input is the same; otherwise, it overwrites the last dimension
  ) {
    auto* t = new Tensor(name, layer_id, input->dtype, TensorType::TENSOR_TYPE_ACTIVATION,
                         OperationType::OP_TYPE_SSM_SCAN);
    std::ranges::copy(input->dims, t->dims.begin());

    if (output_inner_size > 0) {
      for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (t->dims[i] != 0) {
          t->dims[i] = output_inner_size;
          break;
        }
      }
    }

    t->src[0] = input;
    t->src[1] = weightPlaceHolder(a_info, a_info->name, layer_id);
    t->src[2] = weightPlaceHolder(alpha_info, alpha_info->name, layer_id);
    t->src[3] = weightPlaceHolder(beta_info, beta_info->name, layer_id);
    t->src[4] = weightPlaceHolder(dt_bias_info, dt_bias_info->name, layer_id);

    computeStrides(t);
    return t;
  }
  // ReSharper restore CppDFAMemoryLeak
};

#endif //OP_FACTORY_HPP
