#include "codec.h"

#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

std::string dtype_to_string(DType dtype) {
	switch (dtype) {
	case DType::F32:
		return "F32";
	case DType::F16:
		return "F16";
	case DType::BF16:
		return "BF16";
	case DType::F8E5M2:
		return "F8_E5M2";
	case DType::F8E4M3:
		return "F8_E4M3";
	case DType::I32:
		return "I32";
	case DType::I16:
		return "I16";
	case DType::I8:
		return "I8";
	case DType::U8:
		return "U8";
	}
	return "UNKNOWN";
}

size_t dtype_size(DType dtype) {
	switch (dtype) {
	case DType::F32:
		return 4;
	case DType::F16:
		return 2;
	case DType::BF16:
		return 2;
	case DType::F8E5M2:
		return 1;
	case DType::F8E4M3:
		return 1;
	case DType::I32:
		return 4;
	case DType::I16:
		return 2;
	case DType::I8:
		return 1;
	case DType::U8:
		return 1;
	}
	return 0;
}

int Tensor::from_json(const std::string &name, const json &val, void *bytes_ptr, size_t bytes_size) {
	this->name = name;
	std::string dtype_str = val.value("dtype", "");
	if (dtype_str == "F32") {
		this->dtype = DType::F32;
	} else if (dtype_str == "F16") {
		this->dtype = DType::F16;
	} else if (dtype_str == "BF16") {
		this->dtype = DType::BF16;
	} else if (dtype_str == "F8_E5M2") {
		this->dtype = DType::F8E5M2;
	} else if (dtype_str == "F8_E4M3") {
		this->dtype = DType::F8E4M3;
	} else if (dtype_str == "I32") {
		this->dtype = DType::I32;
	} else if (dtype_str == "I16") {
		this->dtype = DType::I16;
	} else if (dtype_str == "I8") {
		this->dtype = DType::I8;
	} else if (dtype_str == "U8") {
		this->dtype = DType::U8;
	} else {
		std::cerr << "bad dtype" << std::endl;
		return -1;
	}
	size_t dsize = dtype_size(this->dtype);

	size_t numel = 1;
	if (val.at("shape").size() > 4) {
		std::cerr << "shape exceeds 4 dimensions" << std::endl;
	}
	for (size_t i = 0; i < val.at("shape").size() && i < 4; i++) {
		if (val.at("shape")[i].get<int>() != val.at("shape")[i]) {
			std::cerr << "bad shape" << std::endl;
			return -1;
		}
		shape[i] = val.at("shape")[i].get<int>();
		numel *= shape[i];
	}
	if (val.at("data_offsets").size() != 2) {
		return -1;
	}
	size_t offset_start = static_cast<size_t>(val.at("data_offsets")[0]);
	size_t offset_end = static_cast<size_t>(val.at("data_offsets")[1]);
	if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
		std::cerr << "bad offsets" << std::endl;
		return -1;
	}
	this->data = (char *)bytes_ptr + offset_start;
	this->size = offset_end - offset_start;
	// validate the shape matches the size
	if (numel * dsize != this->size) {
		std::cerr << "bad size" << std::endl;
		return -1;
	}
	return 0;
}

int YALMData::from_file(const std::string &filename) {
	std::cout << "loading data from file: " << filename << std::endl;
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

	uint64_t json_size = *(uint64_t *)data;
	if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
		munmap(data, size);
		return -1;
	}

	char *json_ptr = (char *)data + sizeof(uint64_t);
	void *bytes_ptr = (char *)data + sizeof(uint64_t) + json_size;
	size_t bytes_size = size - sizeof(uint64_t) - json_size;

	std::string json_str(json_ptr, json_size);
	json header = json::parse(json_str);

	for (auto &[key, val] : header.items()) {
		if (key == "__metadata__") {
			metadata = val;
		} else {
			Tensor &tensor = tensors[key];
			if (tensor.from_json(key, val, bytes_ptr, bytes_size) != 0) {
				munmap(data, size);
				return -1;
			}
		}
	}

	return 0;
}