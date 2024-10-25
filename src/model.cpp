#include "model.h"

#include "json.hpp"
#include <algorithm>
#include <cfloat>
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
}