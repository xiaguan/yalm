#pragma once

#include "model.h"

struct Sampler {
  int vocab_size;

  Sampler(const Config& config);

  int sample_argmax(const float* logits);
};