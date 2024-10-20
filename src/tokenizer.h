#pragma once

#include "codec.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

struct TokenTrie {
  std::unordered_map<char, TokenTrie> children;
  // If non-negative, then this represents the ID of the token formed by the path from the root to this node.
  int token_id = -1;
};

/*
A tokenizer vocab will look like this in the metadata of a .yalm file:
```
"tokenizer.tokens": [
  "<unk>",        // 0
  "<s>",          // 1
  "</s>",         // 2
  "<0x00>",       // 3--------------+
  "<0x01>",       // 4              |  Byte
  "<0x02>",       // 5              |  Fallback 
  ...                               |  Tokens
  "<0xFE>",       // 257            |
  "<0xFF>",       // 258------------+
  "▁▁",           // 259
  "▁▁▁▁",         // 260
  "▁t",           // 261
  "in",           // 262
  "er",           // 263
  ...
],
"tokenizer.scores": [...]
```
*/

struct Tokenizer {
  // vector where the index is the token id and the value is the token string
  std::vector<std::string> vocab;
  // trie mapping token strings to token ids
  TokenTrie vocab_trie;

  int bos_id = -1;
  int eos_id = -1;
  int eot_id = -1;
  // start index of the byte fallback range (256 tokens). -1 if none.
  int byte_fallback_start = 0;

  // convenience array containing the decodings for the fixed 256 byte fallbacks '{0x00}\0', '{0x01}\0', ..., '{0xFF}\0'.
  // TODO: use constexpr?
  std::string byte_pieces[256];

  Tokenizer(const YALMData& data, int bos_id, int eos_id);

  std::vector<int> encode(const std::string& text) const;
  std::string decode_one(int prev_token, int token) const;
};