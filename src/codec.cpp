#include "codec.h"

#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>

int Tensor::from_json(const std::string& name, const json& val, void* bytes_ptr) {
  tensor.name = name;
  size_t dsize = 0;
  switch (val.value("dtype", "").get<std::string>()) {
    case "F32": {
      tensor.dtype = DType::dt_f32;
      dsize = 4;
      break;
    }
    case "F16": {
      tensor.dtype = DType::dt_f16;
      dsize = 2;
      break;
    }
    case "BF16": {
      tensor.dtype = DType::dt_bf16;
      dsize = 2;
      break;
    }
    case "F8_E5M2": {
      tensor.dtype = DType::dt_f8e5m2;
      dsize = 1;
      break;
    }
    case "F8_E4M3": {
      tensor.dtype = DType::dt_f8e4m3;
      dsize = 1;
      break;
    }
    case "I32": {
      tensor.dtype = DType::dt_i32;
      dsize = 4;
      break;
    }
    case "I16": {
      tensor.dtype = DType::dt_i16;
      dsize = 2;
      break;
    }
    case "I8": {
      tensor.dtype = DType::dt_i8;
      dsize = 1;
      break;
    }
    case "U8": {
      tensor.dtype = DType::dt_u8;
      dsize = 1;
      break;
    }
    default: {
      std::cout << "bad dtype" << std::endl;
      return -1;
    }
  }

  size_t numel = 1;
  for (size_t i = 0; i < val.at("shape").size() && i < 8; i++) {
    tensor.shape[i] = val.at("shape")[i].get<int>();
    numel *= tensor.shape[i];
  }
  if (val.at("offsets").size() != 2) {
    return -1;
  }
  int offset_start = val.at("offsets")[0].get<int>();
  int offset_end = val.at("offsets")[1].get<int>();
  if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
    std::cout << "bad offsets" << std::endl;
    return -1;
  }
  tensor->data = (char*)bytes_ptr + offset_start;
  tensor->size = offset_end - offset_start;
  // validate the shape matches the size
  if (numel * dsize != tensor->size) {
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
      if (tensor.from_json(key, val, bytes_ptr) != 0) {
        munmap(data, size);
        return -1;
      }
    }
  }

  return 0;
}