#ifndef QWEN3_H
#define QWEN3_H
#include "model.h"

// Qwen3 model structure parameters (parsed from GGUF metadata)
struct Qwen3Config {
  int hidden_size = 1024;
  int num_layers = 28;
  int num_heads = 16;
  int num_kv_heads = 8;
  int head_dim = 128;
  int intermediate_size = 3072;
  int vocab_size = 0;
  int max_seq_len = 2048;
  float rms_norm_eps = 1e-6F;
  float rope_theta = 1000000.0F;
};

class Qwen3Model final : public ModelBase {
private:
  Qwen3Config config;

  void parseConfig(const GGUFInfo& info);

  [[nodiscard]] Tensor* buildQwen3Layer(
    Tensor* input,
    const GGUFInfo& info,
    int layer_index,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int intermediate_size,
    float rms_norm_eps,
    Tensor* rope_cos,
    Tensor* rope_sin);

public:
  Qwen3Model() {
    name = "Qwen3";
    type = ModelType::CAUSAL_LM;
    arch = ModelArchType::QWEN3;
  }

  ~Qwen3Model() override = default;

  Qwen3Model(const Qwen3Model&) = delete;
  Qwen3Model& operator=(const Qwen3Model&) = delete;
  Qwen3Model(Qwen3Model&&) noexcept = default;
  Qwen3Model& operator=(Qwen3Model&&) noexcept = default;

  [[nodiscard]] const Qwen3Config& getConfig() const { return config; }

  [[nodiscard]] size_t vocabSize() const override { return config.vocab_size; }
  [[nodiscard]] int64_t maxSeqLen() const override { return config.max_seq_len; }

  [[nodiscard]] std::unique_ptr<ComputeGraph> buildGraph(const GGUFInfo&) override;
  void printInfo() override;
};

#endif //QWEN3_H
