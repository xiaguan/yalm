#pragma once

#include "json.hpp"

#include <string>
#include <unordered_map>

using json = nlohmann::json;

// TODO: Should this be narrowed down to what we actually support for model weight representation?
enum class DType {
	dt_f32,
	dt_f16,
	dt_bf16,
	dt_f8e5m2,
	dt_f8e4m3,
	dt_i32,
	dt_i16,
	dt_i8,
	dt_u8,
};

struct Tensor {
  std::string name;
  DType dtype;
  int shape[8] = {}; // Initialize the shape array with zeros
  void* data = nullptr;
  size_t size;

  // Returns 0 if successful, other if failed
  int from_json(const std::string& name, const json& j, void* bytes_ptr, size_t bytes_size);
};

struct YALMData {
  void* data = nullptr;
  size_t size;

  json metadata;

  std::unordered_map<std::string, Tensor> tensors;

  // Initialize a YALMData object from a .yalm file which was created by `convert.py`.
  // Returns 0 if successful, other if failed
  int from_file(const std::string& filename);
};