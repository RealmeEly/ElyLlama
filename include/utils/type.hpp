#ifndef DTYPE_HPP
#define DTYPE_HPP

#include <cstdint>
#include <string>

typedef enum class Device : uint8_t {
  CPU = 0,
  UNKNOWN = 255
} Device;

constexpr auto operator<=>(Device lhs, Device rhs) noexcept {
  return static_cast<uint8_t>(lhs) <=> static_cast<uint8_t>(rhs);
}

typedef enum class TensorType : uint32_t {
  TENSOR_TYPE_UNKNOWN = 0,
  TENSOR_TYPE_WEIGHT = 1,
  TENSOR_TYPE_CACHE = 2,
  TENSOR_TYPE_KV_CACHE = 3,
  TENSOR_TYPE_INPUT = 4,
  TENSOR_TYPE_ACTIVATION = 5,
  TENSOR_TYPE_OUTPUT = 6,
  TENSOR_TYPE_VIEW = 7
} TensorType;

typedef enum class DataType : uint32_t {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  GGML_TYPE_Q4_2 = 4, // support has been removed
  GGML_TYPE_Q4_3 = 5, // support has been removed
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS = 17,
  GGML_TYPE_IQ3_XXS = 18,
  GGML_TYPE_IQ1_S = 19,
  GGML_TYPE_IQ4_NL = 20,
  GGML_TYPE_IQ3_S = 21,
  GGML_TYPE_IQ2_S = 22,
  GGML_TYPE_IQ4_XS = 23,
  GGML_TYPE_I8 = 24,
  GGML_TYPE_I16 = 25,
  GGML_TYPE_I32 = 26,
  GGML_TYPE_I64 = 27,
  GGML_TYPE_F64 = 28,
  GGML_TYPE_IQ1_M = 29,
  GGML_TYPE_BF16 = 30,
  GGML_TYPE_Q4_0_4_4 = 31, // support has been removed from gguf files
  GGML_TYPE_Q4_0_4_8 = 32,
  GGML_TYPE_Q4_0_8_8 = 33,
  GGML_TYPE_TQ1_0 = 34,
  GGML_TYPE_TQ2_0 = 35,
  GGML_TYPE_IQ4_NL_4_4 = 36,
  GGML_TYPE_IQ4_NL_4_8 = 37,
  GGML_TYPE_IQ4_NL_8_8 = 38,
  GGML_TYPE_MXFP4 = 39, // MXFP4 (1 block)
  GGML_TYPE_COUNT = 40,
} DataType;

typedef enum class GGUFType : uint32_t {
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8,
  GGUF_TYPE_ARRAY = 9,
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12,
} GGUFType;

typedef enum class OperationType : uint32_t {
  OP_TYPE_NONE = 0,
  OP_TYPE_MEMCPY = 1,
  OP_TYPE_DUP = 2,
  OP_TYPE_ADD = 3,
  OP_TYPE_SUB = 4,
  OP_TYPE_MUL = 5,
  OP_TYPE_DIV = 6,
  OP_TYPE_SCALE = 7,
  OP_TYPE_MAT_MUL = 8,
  OP_TYPE_TRANSPOSE = 9,
  OP_TYPE_RESHAPE = 10,
  OP_TYPE_PERMUTE = 11,
  OP_TYPE_VIEW = 12,
  OP_TYPE_CONCAT = 13,
  OP_TYPE_REPEAT = 14,
  OP_TYPE_SOFTMAX = 15,
  OP_TYPE_RMS_NORM = 16,
  OP_TYPE_LAYER_NORM = 17,
  OP_TYPE_GELU = 18,
  OP_TYPE_SILU = 19,
  OP_TYPE_RELU = 20,
  OP_TYPE_DIAG_MASK_INF = 21,
  OP_TYPE_POOL_2D = 22,
  OP_TYPE_UPSCALE = 23,
  OP_TYPE_PAD = 24,
  OP_TYPE_UNPAD = 25,
  OP_TYPE_PLACEHOLDER = 26,
  OP_TYPE_EMBEDDING = 27,
  OP_TYPE_LINEAR = 28,
  OP_TYPE_APPLY_ROPE = 29,
  OP_TYPE_SDPA = 30,
  OP_TYPE_TOKENIZE = 31,
  OP_TYPE_ROPE_CACHE = 32,
  OP_TYPE_CONV2D = 33,
  OP_TYPE_FLASH_ATTN = 34,
  OP_TYPE_CAUSAL_CONV1D = 35,
  OP_TYPE_SSM_SCAN = 36,
  OP_TYPE_NARROW = 37,
  OP_TYPE_SIGMOID = 38,
  OP_COUNT = 39
} OperationType;

typedef enum class ModelType : uint8_t {
  UNKNOWN = 0,
  CAUSAL_LM = 1,
  EMBEDDING = 2,
  SEQ2SEQ = 3,
  CLASSIFIER = 4,
} ModelType;

typedef enum class ModelArchType : uint8_t {
  UNKNOWN = 0,
  QWEN3 = 1,
  QWEN35 = 2
} ModelArchType;

inline std::string ggufTypeToString(const GGUFType type) {
  switch (type) {
  case GGUFType::GGUF_TYPE_BOOL:
    return "BOOL";
  case GGUFType::GGUF_TYPE_UINT8:
    return "UINT8";
  case GGUFType::GGUF_TYPE_INT8:
    return "INT8";
  case GGUFType::GGUF_TYPE_UINT16:
    return "UINT16";
  case GGUFType::GGUF_TYPE_INT16:
    return "INT16";
  case GGUFType::GGUF_TYPE_UINT32:
    return "UINT32";
  case GGUFType::GGUF_TYPE_INT32:
    return "INT32";
  case GGUFType::GGUF_TYPE_FLOAT32:
    return "FLOAT32";
  case GGUFType::GGUF_TYPE_UINT64:
    return "UINT64";
  case GGUFType::GGUF_TYPE_INT64:
    return "INT64";
  case GGUFType::GGUF_TYPE_FLOAT64:
    return "FLOAT64";
  case GGUFType::GGUF_TYPE_STRING:
    return "STRING";
  case GGUFType::GGUF_TYPE_ARRAY:
    return "ARRAY";
  default:
    return "UNKNOWN";
  }
}

inline size_t dataTypeSize(const DataType dtype) noexcept {
  switch (dtype) {
  case DataType::GGML_TYPE_COUNT:
    return 0;
  case DataType::GGML_TYPE_I8:
    return 1;
  case DataType::GGML_TYPE_I16:
  case DataType::GGML_TYPE_F16:
  case DataType::GGML_TYPE_BF16:
    return 2;
  case DataType::GGML_TYPE_I32:
  case DataType::GGML_TYPE_F32:
    return 4;
  case DataType::GGML_TYPE_I64:
  case DataType::GGML_TYPE_F64:
    return 8;
  case DataType::GGML_TYPE_Q2_K:    // Q2_K block size
  case DataType::GGML_TYPE_Q3_K:    // Q3_K block size
  case DataType::GGML_TYPE_Q4_K:    // Q4_K block size
  case DataType::GGML_TYPE_Q5_K:    // Q5_K block size
  case DataType::GGML_TYPE_Q6_K:    // Q6_K block size
  case DataType::GGML_TYPE_Q8_K:    // Q8_K block size
  case DataType::GGML_TYPE_IQ2_XXS: // IQ2_XXS block size
  case DataType::GGML_TYPE_IQ2_XS:  // IQ2_XS block size
  case DataType::GGML_TYPE_IQ3_XXS: // IQ3_XXS block size
  case DataType::GGML_TYPE_IQ1_S:   // IQ1_S block size
  case DataType::GGML_TYPE_IQ4_NL:  // IQ4_NL block size
  case DataType::GGML_TYPE_IQ3_S:   // IQ3_S block size
  case DataType::GGML_TYPE_IQ2_S:   // IQ2_S block size
  case DataType::GGML_TYPE_IQ4_XS:  // IQ4_XS block size
  case DataType::GGML_TYPE_IQ1_M:   // IQ1_M block size
  case DataType::GGML_TYPE_TQ1_0:   // TQ1_0 block size
  case DataType::GGML_TYPE_TQ2_0:   // TQ2_0 block size
  case DataType::GGML_TYPE_MXFP4:
    return 256; // MXFP4 block size
  case DataType::GGML_TYPE_Q4_0:
    return sizeof(float) + 4; // super-block + scale
  case DataType::GGML_TYPE_Q4_1:
    return 2 * sizeof(float) + 4; // super-block + scales
  case DataType::GGML_TYPE_Q5_0:
    return sizeof(float) + 6; // super-block + scale
  case DataType::GGML_TYPE_Q5_1:
    return 2 * sizeof(float) + 6; // super-block + scales
  case DataType::GGML_TYPE_Q8_0:
    return sizeof(float) + 8; // super-block + scale
  case DataType::GGML_TYPE_Q8_1:
    return 2 * sizeof(float) + 8; // super-block + scales
  default:
    return 0;
  }
}

inline std::string operationTypeToString(const OperationType op) {
  switch (op) {
  case OperationType::OP_TYPE_NONE:
    return "None";
  case OperationType::OP_TYPE_DUP:
    return "Dup";
  case OperationType::OP_TYPE_ADD:
    return "Add";
  case OperationType::OP_TYPE_SUB:
    return "Sub";
  case OperationType::OP_TYPE_MUL:
    return "Mul";
  case OperationType::OP_TYPE_DIV:
    return "Div";
  case OperationType::OP_TYPE_SCALE:
    return "Scale";
  case OperationType::OP_TYPE_MAT_MUL:
    return "MatMul";
  case OperationType::OP_TYPE_TRANSPOSE:
    return "Transpose";
  case OperationType::OP_TYPE_RESHAPE:
    return "Reshape";
  case OperationType::OP_TYPE_PERMUTE:
    return "Permute";
  case OperationType::OP_TYPE_VIEW:
    return "View";
  case OperationType::OP_TYPE_CONCAT:
    return "Concat";
  case OperationType::OP_TYPE_REPEAT:
    return "Repeat";
  case OperationType::OP_TYPE_SOFTMAX:
    return "Softmax";
  case OperationType::OP_TYPE_RMS_NORM:
    return "RMSNorm";
  case OperationType::OP_TYPE_LAYER_NORM:
    return "LayerNorm";
  case OperationType::OP_TYPE_GELU:
    return "GELU";
  case OperationType::OP_TYPE_SILU:
    return "SiLU";
  case OperationType::OP_TYPE_RELU:
    return "ReLU";
  case OperationType::OP_TYPE_DIAG_MASK_INF:
    return "DiagMaskInf";
  case OperationType::OP_TYPE_POOL_2D:
    return "Pool2D";
  case OperationType::OP_TYPE_UPSCALE:
    return "Upscale";
  case OperationType::OP_TYPE_PAD:
    return "Pad";
  case OperationType::OP_TYPE_UNPAD:
    return "Unpad";
  case OperationType::OP_TYPE_MEMCPY:
    return "Memcpy";
  case OperationType::OP_TYPE_PLACEHOLDER:
    return "Placeholder";
  case OperationType::OP_TYPE_EMBEDDING:
    return "Embedding";
  case OperationType::OP_TYPE_LINEAR:
    return "Linear";
  case OperationType::OP_TYPE_APPLY_ROPE:
    return "Rope";
  case OperationType::OP_TYPE_SDPA:
    return "SDPA";
  case OperationType::OP_TYPE_TOKENIZE:
    return "Tokenize";
  case OperationType::OP_TYPE_ROPE_CACHE:
    return "RopeCache";
  case OperationType::OP_TYPE_CONV2D:
    return "Conv2D";
  case OperationType::OP_TYPE_FLASH_ATTN:
    return "FlashAttn";
  case OperationType::OP_TYPE_CAUSAL_CONV1D:
    return "CausalConv1D";
  case OperationType::OP_TYPE_SSM_SCAN:
    return "SSMScan";
  case OperationType::OP_TYPE_NARROW:
    return "Narrow";
  case OperationType::OP_TYPE_SIGMOID:
    return "Sigmoid";
  default:
    return "Unknown";
  }
}

inline std::string dataTypeToString(const DataType dtype) {
  switch (dtype) {
  case DataType::GGML_TYPE_F32:
    return "F32";
  case DataType::GGML_TYPE_F16:
    return "F16";
  case DataType::GGML_TYPE_Q4_0:
    return "Q4_0";
  case DataType::GGML_TYPE_Q4_1:
    return "Q4_1";
  case DataType::GGML_TYPE_Q5_0:
    return "Q5_0";
  case DataType::GGML_TYPE_Q5_1:
    return "Q5_1";
  case DataType::GGML_TYPE_Q8_0:
    return "Q8_0";
  case DataType::GGML_TYPE_Q8_1:
    return "Q8_1";
  case DataType::GGML_TYPE_Q2_K:
    return "Q2_K";
  case DataType::GGML_TYPE_Q3_K:
    return "Q3_K";
  case DataType::GGML_TYPE_Q4_K:
    return "Q4_K";
  case DataType::GGML_TYPE_Q5_K:
    return "Q5_K";
  case DataType::GGML_TYPE_Q6_K:
    return "Q6_K";
  case DataType::GGML_TYPE_Q8_K:
    return "Q8_K";
  case DataType::GGML_TYPE_IQ2_XXS:
    return "IQ2_XXS";
  case DataType::GGML_TYPE_IQ2_XS:
    return "IQ2_XS";
  case DataType::GGML_TYPE_IQ3_XXS:
    return "IQ3_XXS";
  case DataType::GGML_TYPE_IQ1_S:
    return "IQ1_S";
  case DataType::GGML_TYPE_IQ4_NL:
    return "IQ4_NL";
  case DataType::GGML_TYPE_IQ3_S:
    return "IQ3_S";
  case DataType::GGML_TYPE_IQ2_S:
    return "IQ2_S";
  case DataType::GGML_TYPE_IQ4_XS:
    return "IQ4_XS";
  case DataType::GGML_TYPE_I8:
    return "I8";
  case DataType::GGML_TYPE_I16:
    return "I16";
  case DataType::GGML_TYPE_I32:
    return "I32";
  case DataType::GGML_TYPE_I64:
    return "I64";
  case DataType::GGML_TYPE_F64:
    return "F64";
  case DataType::GGML_TYPE_IQ1_M:
    return "IQ1_M";
  case DataType::GGML_TYPE_BF16:
    return "BF16";
  case DataType::GGML_TYPE_TQ1_0:
    return "TQ1_0";
  case DataType::GGML_TYPE_TQ2_0:
    return "TQ2_0";
  case DataType::GGML_TYPE_MXFP4:
    return "MXFP4";
  case DataType::GGML_TYPE_COUNT:
    return "COUNT";
  default:
    return "Unknown";
  }
}

inline std::string modelArchToString(const ModelArchType arch) {
  switch (arch) {
  case ModelArchType::QWEN3:
    return "Qwen3";
  case ModelArchType::QWEN35:
    return "Qwen35";
  default:
    return "Unknown";
  }
}

inline std::string modelTypeToString(const ModelType type) {
  switch (type) {
  case ModelType::CAUSAL_LM:
    return "CausalLM";
  case ModelType::EMBEDDING:
    return "Embedding";
  case ModelType::SEQ2SEQ:
    return "Seq2Seq";
  case ModelType::CLASSIFIER:
    return "Classifier";
  default:
    return "Unknown";
  }
}

inline std::string deviceToString(const Device dev) {
  switch (dev) {
  case Device::CPU:
    return "CPU";
  default:
    return "Unknown";
  }
}

#endif //DTYPE_HPP
