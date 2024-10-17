#include "tokenizer.h"

#include <string_view>

constexpr int MAX_TOKEN_LENGTH = 512;
constexpr int MAX_UTF8_BYTES = 4;

std::string Tokenizer::decode_one(int prev_token, int token) const {
  const std::string& piece = vocab[token];
  // if following BOS token, sentencepiece decoder strips any leading whitespace
  if (prev_token == bos_id && piece[0] == ' ') {
    return piece.substr(1);
  }
  // return byte piece for byte fallback tokens (<0x00>, <0x01>, ..., <0xFF>)
  if (byte_fallback_start >= 0 && token >= byte_fallback_start && (token - byte_fallback_start) < 256) {
    return byte_pieces[token - byte_fallback_start];
  }
  return piece;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> out_tokens;
  // TODO: handle BOS token (pass optional flag)

  // 1. process the raw UTF-8 bytes of the input string first, building
  //    a list of token IDs corresponding to byte fallbacks and un-merged tokens.
  for (int i = 0; i < text.size();) {
    if (i + 3 <= text.size() && text[i] == '<' && text[i + 1] == '|') {
      // special token, skip until '|>'
      int l = 3;
      while (i+l < text.size() && l < MAX_TOKEN_LENGTH && text[i + l - 2] != '|' && text[i + l - 1] != '>') {
        l++;
      }
      std::string_view byte_piece(&text[i], l);
      if (byte_piece[l - 2] == '|' && byte_piece[l - 1] == '>') {
        // we found the end of a special token, try to encode it as is
        auto it = vocab_map.find(byte_piece);
        if (it != vocab_map.end()) {
          out_tokens.push_back(it->second);
          i += l;
          continue;
        }
      }
    }

    int l = 1;
    // this byte is a leading byte (11...), so it's a multi-byte UTF8 codepoint
    if ((text[i] & 0xC0) == 0xC0) {
      for (int i = 1; i < MAX_UTF8_BYTES && (text[i] & 0xC0) == 0x80; i++) {
        // if the next byte is a continuation byte (10...), append it to the current codepoint
        l++;
      }
    }
    std::string_view byte_piece(&text[i], l);
    auto it = vocab_map.find(byte_piece);
    if (it != vocab_map.end()) {
      out_tokens.push_back(it->second);
      i += l;
      continue;
    } else if (byte_fallback_start >= 0) {
      // encode using byte fallback
      for (int j = 0; j < l; j++) {
        out_tokens.push_back(byte_fallback_start + static_cast<unsigned char>(byte_piece[j]));
      }
      i += l;
      continue;
    } else {
      // TODO: should we be adding <unk> here?
      i += l;
      continue;
    }
  }

  // 2. merge the tokens.
  // TODO: implement me

  // TODO: handle EOS token (pass optional flag)

  return out_tokens;
}