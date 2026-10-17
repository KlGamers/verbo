#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kernels.hpp"
#include "verbo/verbo.hpp"

namespace verbo {

struct Mapping {
  void* ptr = nullptr;
  std::size_t len = 0;

  explicit Mapping(const std::string& path);
  ~Mapping();
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;

  const std::uint8_t* data() const { return static_cast<const std::uint8_t*>(ptr); }
};

struct Buf {
  std::uint8_t* p = nullptr;
  explicit Buf(std::size_t n);
  ~Buf();
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

struct Layer {
  const float* an = nullptr;
  const float* fn = nullptr;
  const float* qkv_b = nullptr;
  Tensor qkv;
  Tensor o;
  Tensor gu;
  Tensor down;
};

struct Defaults {
  float temp = 0.7f;
  float top_p = 0.9f;
  int top_k = 40;
  float repeat = 1.0f;
};

struct Model::Impl {
  Config cfg;
  Tokenizer tok;
  DType dt = DType::Q8;
  Tensor emb;
  Tensor head;
  const float* norm = nullptr;
  std::vector<Layer> lay;
  int bos = -1;
  std::vector<int> stops;
  std::string chat_kind;
  std::string sys;
  Defaults dflt;
  std::vector<std::unique_ptr<Buf>> own;
  std::vector<std::shared_ptr<Mapping>> maps;

  std::uint8_t* alloc(std::size_t n);
};

}
