#include <iostream>
#include <stdio.h>

#include "codec.h"
#include "model.h"
#include "sampler.h"
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

  std::vector<int> encoding = tokenizer.encode(prompt, true);
  std::string token_encoding_debug_str = "";
  for (int token_id : encoding) {
    if (token_id == tokenizer.bos_id) {
      token_encoding_debug_str += "[<s>:" + std::to_string(token_id) + "]";
    } else if (token_id == tokenizer.eos_id) {
      token_encoding_debug_str += "[</s>:" + std::to_string(token_id) + "]";
    } else {
      token_encoding_debug_str += "[" + tokenizer.vocab[token_id] + ":" + std::to_string(token_id) + "]";
    }
  }
  std::cout << token_encoding_debug_str << std::endl;

  // Hydrate KV cache by forwarding model on all prompt tokens and discarding output.
  // This also generates output logits for the last token.
  for (int pos = 0; pos < encoding.size(); pos++) {
    int token_id = encoding[pos];
    forward(state, model, token_id, pos);
  }
  // For N steps:
  // - Sample + decode output logits
  // - Forward the model
  for (int i = 0; i < num_steps; i++) {
    int token_id = sampler.sample_argmax(state.logits);
    std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
    std::cout << token_str;
    encoding.push_back(token_id);
    forward(state, model, token_id, encoding.size() - 1);
  }

  return 0;
}