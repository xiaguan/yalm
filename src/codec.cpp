#include "codec.h"

#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int Tensor::from_json(const std::string& name, const json& val, void* bytes_ptr, size_t bytes_size) {
  this->name = name;
  size_t dsize = 0;
  std::string dtype_str = val.value("dtype", ""); 
  if (dtype_str == "F32") {
    this->dtype = DType::dt_f32;
    dsize = 4;
  } else if (dtype_str == "F16") {
    this->dtype = DType::dt_f16;
    dsize = 2;
  } else if (dtype_str == "BF16") {
    this->dtype = DType::dt_bf16;
    dsize = 2;
  } else if (dtype_str == "F8_E5M2") {
    this->dtype = DType::dt_f8e5m2;
    dsize = 1;
  } else if (dtype_str == "F8_E4M3") {
    this->dtype = DType::dt_f8e4m3;
    dsize = 1;
  } else if (dtype_str == "I32") {
    this->dtype = DType::dt_i32;
    dsize = 4;
  } else if (dtype_str == "I16") {
    this->dtype = DType::dt_i16;
    dsize = 2;
  } else if (dtype_str == "I8") {
    this->dtype = DType::dt_i8;
    dsize = 1;
  } else if (dtype_str == "U8") {
    this->dtype = DType::dt_u8;
    dsize = 1;
  } else {
    std::cout << "bad dtype" << std::endl;
    return -1;
  }

  size_t numel = 1;
  for (size_t i = 0; i < val.at("shape").size() && i < 8; i++) {
    if (val.at("shape")[i].get<int>() != val.at("shape")[i]) {
      std::cout << "bad shape" << std::endl;
      return -1;
    }
    this->shape[i] = val.at("shape")[i].get<int>();
    numel *= this->shape[i];
  }
  if (val.at("data_offsets").size() != 2) {
    return -1;
  }
  size_t offset_start = static_cast<size_t>(val.at("data_offsets")[0]);
  size_t offset_end = static_cast<size_t>(val.at("data_offsets")[1]);
  if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
    std::cout << "bad offsets" << std::endl;
    return -1;
  }
  this->data = (char*)bytes_ptr + offset_start;
  this->size = offset_end - offset_start;
  // validate the shape matches the size
  if (numel * dsize != this->size) {
    std::cout << "bad size" << std::endl;
    return -1;
  }
  return 0;
}

int YALMData::from_file(const std::string& filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    return -1;
  }

  struct stat st;
	if (fstat(fd, &st) != 0) {
		close(fd);
		return -1;
	}
  
  size = st.st_size;
  data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) {
    close(fd);
    return -1;
  }

#ifdef __linux__
	// increases readahead buffer size, resulting in faster cold loads
	posix_fadvise(fd, 0, size, POSIX_FADV_SEQUENTIAL);
#endif

  close(fd); // fd can be closed after mmap returns without invalidating the mapping

  // Parse the metadata JSON and the tensors
  if (size < sizeof(uint64_t)) {
    munmap(data, size);
    return -1;
  }

  uint64_t json_size = *(uint64_t*)data;
  if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
    munmap(data, size);
    return -1;
  }

  char* json_ptr = (char*)data + sizeof(uint64_t);
	void* bytes_ptr = (char*)data + sizeof(uint64_t) + json_size;
  size_t bytes_size = size - sizeof(uint64_t) - json_size;

  json_ptr[json_size - 1] = 0; // null-terminate the JSON string
  json header = json::parse(json_ptr);

  for (auto& [key, val] : header.items()) {
    if (key == "__metadata__") {
      metadata = val;
    } else {
      std::cout << "parsing tensor: " << key << std::endl;
      if (n_tensors >= MAX_TENSORS) {
        munmap(data, size);
        return -1;
      }
      Tensor& tensor = tensors[n_tensors++];
      if (tensor.from_json(key, val, bytes_ptr, bytes_size) != 0) {
        munmap(data, size);
        return -1;
      }
    }
  }

  return 0;
}