#pragma once

#include "codec.h"

#include <vector>

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
  int n_kv_heads;           // number of key and value heads; can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int vocab_size;           // vocabulary size
  int max_seq_len;          // max sequence length
  float rope_theta;         // RoPE theta
  int rotary_dim;           // dimension of rotary position encoding (elements after that don't get rotated)
  float norm_eps;           // epsilon for layer normalization
  ActivationType act;       // activation function
  LayerNormType norm_type;  // norm type
  float qkv_clip;           // clip qkv values to [-clip, clip]

  // Number of bits per weight, e.g. 8 bits for fp8.
  // Determines type of void* used to store weights.
  // We don't currently support sub-byte weights, but this can be used for e.g. 4-bit gf4 format later.
  int weight_dbits;
  // Data type of the weights according to config, used
  // to safety check tensor dtype at initialization time.
  DType weight_dtype;

  // If nonzero `context` is supplied, max sequence length is limited to `context`.
  void from_yalm(YALMData& yalm, int context = 0);
};

struct Block {
  /* Transformer Block */

  // weights for norms
	float* rms_att_weight = nullptr; // (dim) rmsnorm weights
	float* rms_ffn_weight = nullptr; // (dim)

  // weights for self-attention matmuls
	void* wq = nullptr; // (n_heads * head_dim, dim)
	void* wk = nullptr; // (n_kv_heads * head_dim, dim)
	void* wv = nullptr; // (n_kv_heads * head_dim, dim)
	void* wo = nullptr; // (dim, n_heads * head_dim)
	
  // weights for ffn
	void* w1 = nullptr; // (n_experts?, hidden_dim, dim)
	void* w2 = nullptr; // (n_experts?, dim, hidden_dim)
	void* w3 = nullptr; // (n_experts?, hidden_dim, dim) - GLU weights

  // kv cache
	float* key_cache = nullptr;   // (seq_len, n_kv_heads * head_dim)
	float* value_cache = nullptr; // (seq_len, n_kv_heads * head_dim)

  Block(
    const Config& config,
    const Tensor* rms_att_weight,
    const Tensor* rms_ffn_weight,
    const Tensor* wq,
    const Tensor* wk,
    const Tensor* wv,
    const Tensor* wo,
    const Tensor* w1,
    const Tensor* w2,
    const Tensor* w3
  );
  ~Block();
};

// Buffer for all state used during a forward pass.
// Members are reused across subsequent blocks and passes.
// This lets us avoid allocations during inference.
struct InferenceState {
  // current activations
  float* x = nullptr;         // (dim,) - latest activation
  float* xb = nullptr;        // (dim,) - activation inside a residual branch
  // TODO: do we need xb2?
  float* xb2 = nullptr;       // (dim,) - activation inside a residual branch (second slot)
  float* hb = nullptr;        // (hidden_dim,) - buffer for hidden dimension in feedforward network
  float* hb2 = nullptr;       // (hidden_dim,) - buffer for hidden dimension in feedforward network (second slot)
  float* q = nullptr;         // (n_heads * head_dim,) - query vectors for latest timestamp
  float* k = nullptr;         // (n_kv_heads * head_dim,) - key vectors for latest timestamp
  float* v = nullptr;         // (n_kv_heads * head_dim,) - value vectors for latest timestamp
  float* att = nullptr;       // (n_heads, seq_len) - buffer for attention scores
  // LM head
  float* logits = nullptr;    // (vocab_size,) - final output logits

  InferenceState(const Config& config);
  ~InferenceState();
};

struct Model {
  Config config;

  std::vector<Block> blocks;
  
  // token embedding table
	void* token_embedding_table = nullptr; // (vocab_size, dim)
  // final norm
	float* rms_final_weight = nullptr; // (dim,)
	// classifier weights for the logits, on the last layer
	void* wcls = nullptr; // (dim, vocab_size)

  Model(YALMData& yalm);
};

void forward(InferenceState& s, Model& m, int token, int pos);