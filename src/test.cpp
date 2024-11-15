#include <iostream>
#include <vector>

#include "model.h"

bool floatEquals(float a, float b, float epsilon = 1e-5) {
  return std::abs(a - b) < epsilon;
}

bool arrayEquals(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (!floatEquals(a[i], b[i], epsilon)) {
      return false;
    }
  }
  return true;
}

void assertArrayEquals(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& message) {
  if (!arrayEquals(actual, expected)) {
    std::cerr << "Assertion failed: " << message << std::endl;
    std::cerr << "actual: ";
    for (size_t i = 0; i < actual.size(); i++) {
      std::cerr << actual[i] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "expected: ";
    for (size_t i = 0; i < expected.size(); i++) {
      std::cerr << expected[i] << " ";
    }
    std::cerr << std::endl;
    exit(1);
  }
}

void assertArrayEquals(float* actual, const std::vector<float>& expected, const std::string& message) {
  std::vector<float> actual_array;
  for (size_t i = 0; i < expected.size(); i++) {
    actual_array.push_back(actual[i]);
  }
  assertArrayEquals(actual_array, expected, message);
}

void test_attn() {
  constexpr int TEST_SEQ_LEN = 4;
  constexpr int TEST_DIM = 6;
  constexpr int TEST_HEAD_DIM = 3;
  constexpr int TEST_N_HEADS = 2;
  constexpr int TEST_N_KV_HEADS = 1;
  InferenceState s(
    TEST_DIM,
    TEST_DIM,
    TEST_HEAD_DIM,
    TEST_N_HEADS,
    TEST_N_KV_HEADS,
    1,
    TEST_SEQ_LEN
  );
  // (n_heads, head_dim) - query vectors
  std::vector<float> q{
    0., 1e4, 0., // h=0
    0., 0., 1e4 // h=1
  };
  for (size_t i = 0; i < q.size(); i++) {
    s.q()[i] = q[i];
  }
  std::vector<float> kb{
    1., 0., 0., // t=0
    0., 1., 0., // t=1
    0., 0., 1., // t=2
    -1., 0., 0. // t=3
  }; // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  std::vector<float> vb{
    1., 0., 0., // t=0
    0., 1., 0., // t=1
    0., 0., 1., // t=2
    -1., 0., 0. // t=3
  }; // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads

  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = TEST_N_HEADS / TEST_N_KV_HEADS; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < TEST_N_HEADS; h++) {
    int kv_head_offset = (h / q_per_kv_head) * TEST_HEAD_DIM;
    float* kh = kb.data() + kv_head_offset;
    float* vh = vb.data() + kv_head_offset;
    attn(s.xb(h), s.att(h), s.q(h), kh, vh, TEST_HEAD_DIM, TEST_N_KV_HEADS, TEST_SEQ_LEN);
  }
  // attention scores
  // h=0
  assertArrayEquals(s.att(0), {
    0., 1., 0., 0.
  }, "att(h=0)");
  // h=1
  assertArrayEquals(s.att(1), {
    0., 0., 1., 0.
  }, "att(h=1)");
  assertArrayEquals(s.xb(), {
    0., 1., 0., // h=0
    0., 0., 1. // h=1
  }, "xout");
}

int main(int argc, char* argv[]) {
  test_attn();
  std::cout << "All tests passed" << std::endl;
  return 0;
}