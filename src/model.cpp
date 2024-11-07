#include "model.h"

#include "json.hpp"
#include <algorithm>
#include <cfloat>
#include "fmt/format.h"
#include <iostream>
#include <limits.h>
#include <string>

using json = nlohmann::json;

void Config::from_yalm(YALMData& yalm, int context) {
  dim = std::stoi(yalm.metadata.at("dim").get<std::string>());
	hidden_dim = std::stoi(yalm.metadata.at("hidden_dim").get<std::string>());
	head_dim = std::stoi(yalm.metadata.at("head_dim").get<std::string>());
	n_layers = std::stoi(yalm.metadata.at("n_layers").get<std::string>());
	n_heads = std::stoi(yalm.metadata.at("n_heads").get<std::string>());
	n_kv_heads = std::stoi(yalm.metadata.at("n_kv_heads").get<std::string>());
	vocab_size = std::stoi(yalm.metadata.at("vocab_size").get<std::string>());

	// for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly specified
	max_seq_len = std::max(std::stoi(yalm.metadata.at("max_seq_len").get<std::string>()), 4096);
  if (context) {
		max_seq_len = context;
	}

	rope_theta = std::stof(yalm.metadata.at("rope_theta").get<std::string>());
	rotary_dim = std::stoi(yalm.metadata.at("rotary_dim").get<std::string>());

	norm_eps = std::stof(yalm.metadata.value("norm_eps", "1e-5"));

  std::string act_str = yalm.metadata.value("act_type", "gelu");
  if (act_str == "gelu") {
    act = ActivationType::GELU;
  } else if (act_str == "silu") {
    act = ActivationType::SILU;
  } else {
    std::cout << "unsupported act_type, defaulting to gelu" << std::endl;
    act = ActivationType::GELU;
  }

	std::string norm_type_str = yalm.metadata.value("norm_type", "rmsnorm");
  if (norm_type_str == "rmsnorm") {
    norm_type = LayerNormType::RMSNorm;
  } else {
    std::cout << "unsupported norm_type, defaulting to rmsnorm" << std::endl;
    norm_type = LayerNormType::RMSNorm;
  }

	qkv_clip = yalm.metadata.contains("qkv_clip") ? std::stof(yalm.metadata.at("qkv_clip").get<std::string>()) : FLT_MAX;

  std::string dtype = yalm.metadata.at("dtype").get<std::string>();
  if (dtype == "fp32") {
    weight_dbits = 32;
    weight_dtype = dt_f32;
  } else if (dtype == "fp16") {
    weight_dbits = 16;
    weight_dtype = dt_f16;
  } else if (dtype == "fp8") {
    weight_dbits = 8;
    weight_dtype = dt_f8e5m2;
  } else {
    std::cerr << "FATAL: unsupported dtype: " << dtype << std::endl;
    assert(false);
  }
}

void* check_tensor(Tensor* tensor, DType weight_dtype, int[4] shape) {
  if (tensor == nullptr) {
    std::cerr << "FATAL: missing tensor" << std::endl;
    assert(false);
    return nullptr;
  }
  if (tensor->dtype != weight_dtype || memcmp(tensor->shape, shape, 4 * sizeof(int)) != 0) {
    std::cerr << "FATAL: tensor mismatch" << std::endl;
    std::cerr 
      << fmt::format("expected: dtype={}, shape=[{},{},{},{}]", weight_dtype, shape[0], shape[1], shape[2], shape[3]) 
      << std::endl;
    std::cerr 
      << fmt::format("got: dtype={}, shape=[{},{},{},{}]", tensor->dtype, tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3]) 
      << std::endl;
    assert(false);
  }
  return tensor->data;
};

Tensor* get_tensor(const YALMData& yalm, const std::string& key) {
  auto it = yalm.tensors.find(key);
  if (it == yalm.tensors.end()) {
    std::cerr << "FATAL: missing tensor: " << key << std::endl;
    assert(false);
    return nullptr;
  }
  const Tensor& tensor = it->second;
  return &tensor;
};

Block::Block(
  const Config& config,
  Tensor* rms_att_weight,
  Tensor* rms_ffn_weight,
  Tensor* wq,
  Tensor* wk,
  Tensor* wv,
  Tensor* wo,
  Tensor* w1,
  Tensor* w2,
  Tensor* w3,
) {

  this->rms_att_weight = static_cast<float*>(check_tensor(
    rms_att_weight, DType::dt_f32, {config.dim, 0, 0, 0}
  ));
  this->rms_ffn_weight = static_cast<float*>(check_tensor(
    rms_ffn_weight, DType::dt_f32, {config.dim, 0, 0, 0}
  ));

  this->wq = check_tensor(
    wq, config.weight_dtype, {config.n_heads * config.head_dim, config.dim, 0, 0}
  );
  this->wk = check_tensor(
    wk, config.weight_dtype, {config.n_kv_heads * config.head_dim, config.dim, 0, 0}
  );
  this->wv = check_tensor(
    wv, config.weight_dtype, {config.n_kv_heads * config.head_dim, config.dim, 0, 0}
  );
  this->wo = check_tensor(
    wo, config.weight_dtype, {config.dim, config.n_heads * config.head_dim, 0, 0}
  );

  this->w1 = check_tensor(
    w1, config.weight_dtype, {config.hidden_dim, config.dim, 0, 0}
  );
  this->w2 = check_tensor(
    w2, config.weight_dtype, {config.dim, config.hidden_dim, 0, 0}
  );
  this->w3 = check_tensor(
    w3, config.weight_dtype, {config.hidden_dim, config.dim, 0, 0}
  );

  this->key_cache = new float[config.max_seq_len * config.n_kv_heads * config.head_dim]();
  this->value_cache = new float[config.max_seq_len * config.n_kv_heads * config.head_dim]();
}

Block::~Block() {
  delete[] key_cache;
  delete[] value_cache;
}

InferenceState::InferenceState(const Config& config) {
  x = new float[config.dim]();
  xb = new float[config.dim]();
  xb2 = new float[config.dim]();
  hb = new float[config.hidden_dim]();
  hb2 = new float[config.hidden_dim]();
  q = new float[config.n_heads * config.head_dim]();
  k = new float[config.n_kv_heads * config.head_dim]();
  v = new float[config.n_kv_heads * config.head_dim]();
  att = new float[config.n_heads * config.max_seq_len]();
  logits = new float[config.vocab_size]();
}

InferenceState::~InferenceState() {
  delete[] x;
  delete[] xb;
  delete[] xb2;
  delete[] hb;
  delete[] hb2;
  delete[] q;
  delete[] k;
  delete[] v;
  delete[] att;
  delete[] logits;
}

Model::Model(YALMData& yalm) {
  config.from_yalm(yalm);
  std::cout << "loading model with dtype: " << config.weight_dtype << std::endl;

  token_embedding_table = check_tensor(
    get_tensor(yalm, "model.embed.weight"), 
    config.weight_dtype,
    {config.vocab_size, config.dim, 0, 0}
  );

  for (int i = 0; i < config.n_layers; ++i) {
    blocks.emplace_back(
      config,
      get_tensor(yalm, fmt::format("model.layers.{}.attn.norm.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.norm.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wq.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wk.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wv.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wo.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w1.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w2.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w3.weight", i))
    );
  }

  rms_final_weight = check_tensor(
    get_tensor(yalm, "model.norm.weight"), 
    DType::dt_f32, 
    {config.dim, 1, 1, 1}
  );
  wcls = check_tensor(
    get_tensor(yalm, "model.output.weight"), 
    config.weight_dtype, 
    {config.dim, config.vocab_size, 1, 1}
  );
}