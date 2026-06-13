#include "core/gguf_parser.h"

#include <iostream>

size_t TensorInfo::bytes() const {
  const size_t elem_size = dataTypeSize(dtype);
  size_t num_elems = 1;
  for (const int64_t dim : dimensions) {
    num_elems *= dim;
  }
  return elem_size * num_elems;
}

void GGUFInfo::printInfo() const {
  std::println(std::cout, "{:=<14} GGUF file infomation {:=<14}", "", "");
  std::println(std::cout, "gguf version:             {}", version);
  std::println(std::cout, "model arch:               {}", getModelArchitecture());
  std::println(std::cout, "model name:               {}", getModelName());
  std::println(std::cout, "kv_count:                 {}", meta_data_kv_count);
  std::println(std::cout, "tensor_count:             {}", tensor_count);
  std::println(std::cout, "data_offset:              {}", offset);
  std::println(std::cout, "{:-<26} {:-<8} {:-<14}", "", "", "");
  std::println(std::cout, "{:<26} {:<8} {}", "name", "data_type", "dimension");
  std::println(std::cout, "{:-<26} {:-<8} {:-<14}", "", "", "");
  for (const auto& info : tensor_info_vec) {
    std::println(std::cout, "{:<26} {:<8} {}", info.name, dataTypeToString(info.dtype), info.dimensions);
  }
  std::println(std::cout, "{:-<26} {:-<8} {:-<14}", "", "", "");
}

std::string GGUFInfo::getModelName() const {
  if (meta_data.contains("general.name") && meta_data["general.name"].is_string()) {
    return meta_data["general.name"].get<std::string>();
  }
  return "unknown";
}

std::string GGUFInfo::getModelArchitecture() const {
  if (meta_data.contains("general.architecture") && meta_data["general.architecture"].is_string()) {
    return meta_data["general.architecture"].get<std::string>();
  }
  return "unknown";
}

GGUFParser::GGUFParser(const std::string& file_name) {
  file.open(file_name, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + file_name);
  }

  char magic[4];
  if (!file.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    throw std::runtime_error("Invalid GGUF file: magic number mismatch");
  }

  info = parseInfo();
}

GGUFParser::~GGUFParser() {
  if (file.is_open()) {
    file.close();
  }
}

void GGUFParser::readTensorData(uint64_t tensor_offset, void* dst, size_t size) {
  auto abs_offset = tensor_offset + data_offset;
  file.seekg(static_cast<std::streamoff>(abs_offset));
  if (!file) {
    throw std::runtime_error(std::format("GGUFParser: seek to {} (base={}) failed", abs_offset, data_offset));
  }
  file.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
  if (!file) {
    throw std::runtime_error(std::format("GGUFParser: read {} bytes at offset {} failed", size, abs_offset));
  }
}

GGUFInfo GGUFParser::parseInfo() {
  GGUFInfo info;
  info.version = readUint32Len();
  info.tensor_count = readUint64Len();
  info.meta_data_kv_count = readUint64Len();

  if (info.version != 3) {
    throw std::runtime_error(
      "Unsupported GGUF version: " + std::to_string(info.version) + " (only version 3 is supported)");
  }

  info.meta_data = parseMetadata(info.meta_data_kv_count);
  info.tensor_info_vec = parseTensorInfoVector(info.tensor_count);

  uint32_t alignment = 32;
  const uint64_t raw_offset = file.tellg();
  if (info.meta_data.contains("general.alignment")) {
    alignment = info.meta_data["general.alignment"].get<uint32_t>();
  }
  data_offset = raw_offset + (alignment - raw_offset % alignment) % alignment;
  info.offset = data_offset;

  return info;
}

uint8_t GGUFParser::readUint8Len() {
  uint8_t bytes[1];
  if (!file.read(reinterpret_cast<char*>(bytes), 1)) {
    throw std::runtime_error("Failed to read uint8_t from file");
  }
  return bytes[0];
}

uint16_t GGUFParser::readUint16Len() {
  uint8_t bytes[2];
  if (!file.read(reinterpret_cast<char*>(bytes), 2)) {
    throw std::runtime_error("Failed to read uint16_t from file");
  }
  return static_cast<uint16_t>(bytes[0]) | static_cast<uint16_t>(bytes[1]) << 8;
}

uint32_t GGUFParser::readUint32Len() {
  uint8_t bytes[4];
  if (!file.read(reinterpret_cast<char*>(bytes), 4)) {
    throw std::runtime_error("Failed to read uint32_t from file");
  }
  return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 | static_cast<uint32_t>(bytes[2]) <<
         16 | static_cast<uint32_t>(bytes[3]) << 24;
}

uint64_t GGUFParser::readUint64Len() {
  uint8_t bytes[8];
  if (!file.read(reinterpret_cast<char*>(bytes), 8)) {
    throw std::runtime_error("Failed to read uint64_t from file");
  }
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
  }
  return value;
}

float GGUFParser::readFloat32Len() {
  uint8_t bytes[4];
  if (!file.read(reinterpret_cast<char*>(bytes), 4)) {
    throw std::runtime_error("Failed to read float from file");
  }
  const uint32_t value_uint32 = static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 | static_cast<
                                  uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
  return std::bit_cast<float>(value_uint32);
}

double GGUFParser::readFloat64Len() {
  uint8_t bytes[8];
  if (!file.read(reinterpret_cast<char*>(bytes), 8)) {
    throw std::runtime_error("Failed to read double from file");
  }
  uint64_t value_uint64 = 0;
  for (int i = 0; i < 8; ++i) {
    value_uint64 |= static_cast<uint64_t>(bytes[i]) << (i * 8);
  }
  return std::bit_cast<double>(value_uint64);
}

std::string GGUFParser::readString() {
  const uint64_t str_len = readUint64Len();
  std::string str(str_len, '\0');
  if (!file.read(str.data(), static_cast<std::streamsize>(str_len))) {
    throw std::runtime_error("Failed to read string from file");
  }
  return str;
}

Json GGUFParser::readMetadataValue(GGUFType type) { // NOLINT(*-no-recursion)
  switch (type) {
    case GGUFType::GGUF_TYPE_BOOL:
      return readUint8Len() != 0;
    case GGUFType::GGUF_TYPE_UINT8:
      return readUint8Len();
    case GGUFType::GGUF_TYPE_INT8:
      return static_cast<int8_t>(readUint8Len());
    case GGUFType::GGUF_TYPE_UINT16:
      return readUint16Len();
    case GGUFType::GGUF_TYPE_INT16:
      return static_cast<int16_t>(readUint16Len());
    case GGUFType::GGUF_TYPE_UINT32:
      return readUint32Len();
    case GGUFType::GGUF_TYPE_INT32:
      return static_cast<int32_t>(readUint32Len());
    case GGUFType::GGUF_TYPE_FLOAT32:
      return readFloat32Len();
    case GGUFType::GGUF_TYPE_UINT64:
      return readUint64Len();
    case GGUFType::GGUF_TYPE_INT64:
      return static_cast<int64_t>(readUint64Len());
    case GGUFType::GGUF_TYPE_FLOAT64:
      return readFloat64Len();
    case GGUFType::GGUF_TYPE_STRING:
      return readString();
    case GGUFType::GGUF_TYPE_ARRAY: {
      const uint32_t elem_type = readUint32Len();
      const uint64_t arr_len = readUint64Len();
      Json arr = Json::array();
      for (uint64_t i = 0; i < arr_len; ++i) {
        arr.push_back(readMetadataValue(static_cast<GGUFType>(elem_type)));
      }
      return arr;
    }
    default:
      throw std::runtime_error("Unsupported GGUF metadata type: " + ggufTypeToString(type));
  }
}

Json GGUFParser::parseMetadata(uint64_t kv_count) {
  Json meta_data;
  for (uint64_t i = 0; i < kv_count; ++i) {
    const auto key = readString();
    meta_data[key] = readMetadataValue(static_cast<GGUFType>(readUint32Len()));
  }
  return meta_data;
}

std::vector<TensorInfo> GGUFParser::parseTensorInfoVector(uint64_t tensor_count) {
  std::vector<TensorInfo> tensor_info_vec;
  tensor_info_vec.reserve(tensor_count);
  for (uint64_t i = 0; i < tensor_count; ++i) {
    TensorInfo tensor_info;
    tensor_info.name = readString();
    const uint32_t n_dims = readUint32Len();
    tensor_info.dimensions.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      tensor_info.dimensions[n_dims - d - 1] = static_cast<int64_t>(readUint64Len());
    }
    tensor_info.dtype = static_cast<DataType>(readUint32Len());
    tensor_info.offset = readUint64Len();
    tensor_info_vec.push_back(std::move(tensor_info));
  }
  return tensor_info_vec;
}
