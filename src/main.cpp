#include <cstdint>
#include <iostream>
#include <stdio.h>

#include "fmt/format.h"

#include "codec.h"
#include "model.h"
#include "sampler.h"
#include "time.h"
#include "tokenizer.h"

void error_usage() {
	fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
	fprintf(stderr, "Example: run model.yalm -i \"Q: What is the meaning of life?\"\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len, -1 = infinite\n");
	exit(1);
}

int main(int argc, char* argv[]) {
  // default params
  std::string checkpoint_path = "";    // e.g. out/model.bin
  std::string prompt = "";             // prompt string
  int num_steps = 256;                 // number of steps to run for

	if (argc >= 2) {
		checkpoint_path = argv[1];
	} else {
		error_usage();
	}
	for (int i = 2; i < argc;) {
		// do some basic validation
		if (i + 1 >= argc) {
			error_usage();
		} // must have arg after flag
		if (argv[i][0] != '-') {
			error_usage();
		} // must start with dash
		if (strlen(argv[i]) != 2) {
			error_usage();
		} // must be -x (one dash, one letter)

		// read in the args
		if (argv[i][1] == 'i') {
      if (i + 1 >= argc) {
        error_usage();
      }
      prompt = argv[i + 1];
      i += 2;
		} else if (argv[i][1] == 'n') {
      if (i + 1 >= argc) {
        error_usage();
      }
      num_steps = std::stoi(argv[i + 1]);
      i += 2;
    } else {
			error_usage();
		}
	}

  if (prompt.size() == 0) {
    error_usage();
  }

  YALMData model_data;
  model_data.from_file(checkpoint_path);
  Model model(model_data);
  InferenceState state(model.config);
  Sampler sampler(model.config);
  Tokenizer tokenizer(model_data);

  if (num_steps == 0) {
    // `-n 0` means use the full context length
    num_steps = model.config.max_seq_len;
  }

  // Do one inference as warmup.
  // On CPU, this ensures all tensors are loaded into memory via mmap.
  // On GPU, this ensures all tensors are loaded into device memory and 
  // kernels are compiled + instantiated.
  forward(state, model, 0, 0);

  std::vector<int> encoding;
  {
    uint64_t encode_start_ms = get_timestamp_ms();
    encoding = tokenizer.encode(prompt, true);
    uint64_t encode_end_ms = get_timestamp_ms();

    std::cout << tokenizer.encoding_to_debug_string(encoding) << std::endl;
    uint64_t encoding_ms = encode_end_ms - encode_start_ms;
    std::cout << fmt::format(
      "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
      encoding.size(),
      encoding.size() / (encoding_ms / 1000.0),
      (encoding_ms / 1000.0) / encoding.size(),
      encoding_ms / 1000.0
    ) << std::endl;
  }

  uint64_t start_ms = get_timestamp_ms();
  // Hydrate KV cache by forwarding model on all prompt tokens and discarding output.
  // This also generates output logits for the last token.
  for (size_t pos = 0; pos < encoding.size(); pos++) {
    int token_id = encoding[pos];
    forward(state, model, token_id, pos);
  }
  uint64_t end_hydrate_ms = get_timestamp_ms();
  // For N steps:
  // - Sample + decode output logits
  // - Forward the model
  for (int i = 0; i < num_steps || num_steps == -1; i++) {
    int token_id = sampler.sample_argmax(state.logits());
    std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
    std::cout << token_str << std::flush;
    encoding.push_back(token_id);
    if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
      break;
    }
    forward(state, model, token_id, encoding.size() - 1);
  }
  std::cout << "\n" << std::endl;
  uint64_t end_ms = get_timestamp_ms();
  uint64_t elapsed_ms = end_ms - start_ms;
  std::cout << fmt::format(
    "Generation stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, hydrate: {:.5}s, total: {:.5}s)\n",
    encoding.size(),
    encoding.size() / (elapsed_ms / 1000.0),
    (elapsed_ms / 1000.0) / encoding.size(),
    (end_hydrate_ms - start_ms) / 1000.0,
    elapsed_ms / 1000.0
  ) << std::endl;

  return 0;
}