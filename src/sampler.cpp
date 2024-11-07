#include "sampler.h"

#include <cfloat>

Sampler::Sampler(const Config& config) {
  vocab_size = config.vocab_size;
}

int Sampler::sample_argmax(const float* logits) {
  int argmax = 0;
  float max_val = -FLT_MAX;
  for (int i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      argmax = i;
    }
  }
  return argmax;
}