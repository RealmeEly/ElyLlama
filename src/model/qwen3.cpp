#include "model/qwen3.h"

#include <iostream>

void Qwen3Model::parseConfig(const GGUFInfo& info) {
  auto& meta = info.meta_data;
  config.hidden_size = meta.value("qwen3.embedding_length", 1024);
  config.num_layers = meta.value("qwen3.block_count", 28);
  config.num_heads = meta.value("qwen3.attention.head_count", 16);
  config.num_kv_heads = meta.value("qwen3.attention.head_count_kv", 8);
  config.head_dim = meta.value("qwen3.attention.key_length", 128);
  config.intermediate_size = meta.value("qwen3.feed_forward_length", 3072);
  config.max_seq_len = meta.value("qwen3.context_length", 2048);
  config.rms_norm_eps = meta.value("qwen3.attention.layer_norm_rms_epsilon", 1e-6F);
  config.rope_theta = meta.value("qwen3.rope.freq_base", 1000000.0F);

  // get vocab_size from tokenizer.tokens firstly
  if (meta.contains("tokenizer.ggml.tokens") && meta["tokenizer.ggml.tokens"].is_array()) {
    config.vocab_size = static_cast<int>(meta["tokenizer.ggml.tokens"].size());
  }

  if (config.vocab_size == 0) {
    for (const auto& t : info.tensor_info_vec) {
      if (t.name == "token_embd.weight" && t.dimensions.size() == 2) {
        config.vocab_size = static_cast<int>(t.dimensions[1]);
        break;
      }
    }
  }

  if (config.vocab_size == 0) {
    throw std::runtime_error("Cannot determine vocab_size");
  }

  std::println(std::cout, "Config: hidden={}, heads={}, kv_heads={}, head_dim={}, vocab={}, layers={}",
               config.hidden_size, config.num_heads, config.num_kv_heads,
               config.head_dim, config.vocab_size, config.num_layers);
}

Tensor* Qwen3Model::buildQwen3Layer(
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
  Tensor* rope_sin) {
  // TODO: finish it
  return nullptr;
}

std::unique_ptr<ComputeGraph> Qwen3Model::buildGraph(const GGUFInfo&) {
  // TODO: finish it
  return nullptr;
}

void Qwen3Model::printInfo() {
  std::println(std::cout, "{:=<14} Qwen3 Model Info {:=<14}");
  std::println(std::cout, "  Name: {}", name);
  std::println(std::cout, "  Type: {}", modelTypeToString(type));
  std::println(std::cout, "  Architecture: {}", modelArchToString(arch));
  std::println(std::cout, "  hidden_size:  {}", config.hidden_size);
  std::println(std::cout, "  num_layers:   {}", config.num_layers);
  std::println(std::cout, "  num_heads:    {}", config.num_heads);
  std::println(std::cout, "  num_kv_heads: {}", config.num_kv_heads);
  std::println(std::cout, "  head_dim:     {}", config.head_dim);
  std::println(std::cout, "  vocab_size:   {}", config.vocab_size);
  std::println(std::cout, "  max_seq_len:  {}", config.max_seq_len);
}
