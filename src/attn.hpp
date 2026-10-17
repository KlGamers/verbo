#pragma once

#include <cstddef>
#include <vector>

#include "pool.hpp"

namespace verbo {

constexpr int kKvBlk = 64;

struct AttnArgs {
  const float* q;
  int qld;
  const float* kt;
  std::size_t kth;
  const float* v;
  int kvld;
  float* out;
  int old;
  int n;
  int p0;
  int heads;
  int kv_heads;
  int hd;
  float scale;
};

struct AttnScratch {
  std::vector<std::vector<float>> th;
  std::vector<float> part;
};

void attention(const AttnArgs& a, Pool& pool, AttnScratch& sc);
void exp_inplace(float* x, int n);
void silu_mul(const float* g, const float* u, float* out, int n);

}
