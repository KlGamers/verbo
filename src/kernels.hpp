#pragma once

#include <cstddef>
#include <cstdint>

#include "pool.hpp"
#include "verbo/verbo.hpp"

namespace verbo {

constexpr int kBlk = 32;

struct Tensor {
  DType dt = DType::F32;
  int rows = 0;
  int cols = 0;
  const void* q = nullptr;
  const std::uint16_t* s = nullptr;

  std::size_t row_bytes() const;
  std::size_t qbytes() const { return row_bytes() * static_cast<std::size_t>(rows); }
  std::size_t sbytes() const;
};

float h2f(std::uint16_t h);
std::uint16_t f2h(float f);

void quant_act(const float* x, int n, std::int8_t* q, float* s);
void quant_rows(DType dt, const float* x, int rows, int cols, void* q, std::uint16_t* s);
void deq_row(const Tensor& t, int r, float* out);

float dot_f32(const float* a, const float* b, int n);
float dot_q8(const std::int8_t* w, const std::uint16_t* ws, const std::int8_t* x, const float* xs, int nb);
float dot_q4(const std::uint8_t* w, const std::uint16_t* ws, const std::int8_t* x, const float* xs, int nb);

struct Act {
  const float* f = nullptr;
  const std::int8_t* q = nullptr;
  const float* s = nullptr;
  int n = 0;
  int ld = 0;
};

void matmul(const Tensor& w, const Act& x, float* y, int ldy, Pool& pool);

const char* simd_name();

}
