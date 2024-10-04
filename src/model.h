#pragma once

#include "codec.h"

enum class ActivationType {
  GELU,
  SILU,
};

enum class LayerNormType {
  RMSNorm,
};

struct Config {
  int dim;                  // transformer input & output dimension
  int hidden_dim;           // dimension of hidden layer in feedforward network
  int head_dim;             // dimension of each attention head, usually dim / n_heads
  int n_layers;             // number of layers
  int n_heads;              // number of attention query heads
  int n_kv_heads;           // number of key and value heads; can be < n_heads (this is MultiQueryAttention)
  int vocab_size;           // vocabulary size
  int max_seq_len;          // max sequence length
  float rope_theta;         // RoPE theta
  int rotary_dim;           // dimension of rotary position encoding (elements after that don't get rotated)
  float norm_eps;           // epsilon for layer normalization
  ActivationType act;       // activation function
  LayerNormType norm_type;  // norm type
  float qkv_clip;           // clip qkv values to [-clip, clip]

  // If nonzero `context` is supplied, max sequence length is limited to `context`.
  void from_yalm(YALMData& yalm, int context = 0);
};