#ifndef GGUF_PARSER_H
#define GGUF_PARSER_H

#include "utils/type.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace core {
  using Json = nlohmann::ordered_json;

  // Tensor info
  class TensorInfo {
  public:
    // NOLINTBEGIN(*-non-private-member-variables-in-classes)
    bool transpose = false;
    DataType dtype = DataType::GGML_TYPE_F32;
    uint64_t offset = 0; // absolute file offset
    std::string name = "unknown";
    std::vector<int64_t> dimensions;
    // NOLINTEND(*-non-private-member-variables-in-classes)

    [[nodiscard]] size_t bytes() const;
  };

  // GGUF head info
  class GGUFInfo {
  public:
    // NOLINTBEGIN(*-non-private-member-variables-in-classes)
    Json meta_data;
    uint32_t offset = 0; // tensor start address offset
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t meta_data_kv_count = 0;
    std::vector<TensorInfo> tensor_info_vec;
    // NOLINTEND(*-non-private-member-variables-in-classes)

    void printInfo() const;

    [[nodiscard]] std::string getModelName() const;

    [[nodiscard]] std::string getModelArchitecture() const;
  };

  // GGUF Pasrer
  class GGUFParser {
  private:
    GGUFInfo info;
    std::ifstream file;
    uint64_t data_offset = 0;

  public:
    explicit GGUFParser(const std::string& file_name);

    GGUFParser(const GGUFParser&) = delete;

    GGUFParser& operator=(const GGUFParser&) = delete;

    GGUFParser(GGUFParser&&) = delete;

    GGUFParser& operator=(GGUFParser&&) = delete;

    ~GGUFParser();

    GGUFInfo& getInfo() { return info; }

    uint64_t getDataOffset() const { return data_offset; }

    void readTensorData(uint64_t tensor_offset, void* dst, size_t size);

  private:
    GGUFInfo parseInfo();

    uint8_t readUint8Len();

    uint16_t readUint16Len();

    uint32_t readUint32Len();

    uint64_t readUint64Len();

    float readFloat32Len();

    double readFloat64Len();

    std::string readString();

    Json readMetadataValue(GGUFType type);

    Json parseMetadata(uint64_t kv_count);

    std::vector<TensorInfo> parseTensorInfoVector(uint64_t tensor_count);
  };
} // namespace core

#endif // GGUF_PARSER_H
