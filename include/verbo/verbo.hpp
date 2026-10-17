#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace verbo {

enum class DType : std::uint8_t { F32 = 0, Q8 = 1, Q4 = 2 };

const char* dtype_name(DType dt);
DType parse_dtype(std::string_view s);

struct Config {
  std::string arch;
  int dim = 0;
  int hidden = 0;
  int layers = 0;
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int vocab = 0;
  int max_ctx = 0;
  float eps = 1e-5f;
  float theta = 10000.0f;
  bool qkv_bias = false;
  bool tied = false;
  float rope_factor = 0.0f;
  float rope_low = 1.0f;
  float rope_high = 4.0f;
  int rope_orig = 8192;
};

class Tokenizer {
 public:
  static Tokenizer parse(std::string_view json);

  std::vector<int> encode(std::string_view text, bool special = true) const;
  std::string piece(int id) const;
  std::string decode(const std::vector<int>& ids, bool skip_special = false) const;
  int find(std::string_view tok) const;
  bool is_special(int id) const;
  int size() const;
  const std::string& source() const;

  struct Impl;

 private:
  std::shared_ptr<const Impl> impl_;
};

struct Sampling {
  float temp = 0.7f;
  int top_k = 40;
  float top_p = 0.9f;
  float min_p = 0.0f;
  float repeat = 1.0f;
  int repeat_last = 64;
  std::uint64_t seed = 0;
};

struct Msg {
  std::string role;
  std::string text;
};

class Model {
 public:
  static std::shared_ptr<const Model> load(const std::string& path, DType dt = DType::Q8);

  void save(const std::string& path) const;

  const Config& cfg() const;
  const Tokenizer& tok() const;
  DType dtype() const;
  std::size_t weight_bytes() const;
  int bos() const;
  const std::vector<int>& stops() const;
  bool is_stop(int id) const;
  std::string chat(const std::vector<Msg>& msgs, bool gen = true) const;
  Sampling sampling() const;

  struct Impl;
  const Impl& impl() const { return *impl_; }

 private:
  std::shared_ptr<Impl> impl_;
};

class Session {
 public:
  explicit Session(std::shared_ptr<const Model> mdl, int ctx = 0, int threads = 0);
  ~Session();
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;

  const float* eval(const int* toks, int n);
  const float* eval(const std::vector<int>& toks) { return eval(toks.data(), static_cast<int>(toks.size())); }
  std::vector<float> eval_all(const std::vector<int>& toks);

  int pos() const;
  int ctx() const;
  int threads() const;
  void rewind(int pos);
  const Model& model() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class Sampler {
 public:
  explicit Sampler(Sampling cfg = {});

  int pick(const float* logits, int n);
  void accept(int tok);
  void reset();
  const Sampling& cfg() const { return cfg_; }

 private:
  Sampling cfg_;
  std::uint64_t rng_[2];
  std::vector<int> hist_;
  std::vector<std::pair<float, int>> buf_;

  double unit();
};

class Detok {
 public:
  explicit Detok(const Tokenizer& tok) : tok_(&tok) {}

  std::string push(int id);
  std::string flush();

 private:
  const Tokenizer* tok_;
  std::string pend_;
};

struct Stats {
  int prompt = 0;
  int gen = 0;
  double prompt_ms = 0.0;
  double gen_ms = 0.0;
  bool stopped = false;
  std::vector<int> toks;
};

struct GenOpts {
  int max_tokens = 256;
  Sampling smp;
  std::vector<std::string> stop_text;
};

Stats generate(Session& ses, const std::vector<int>& prompt, const GenOpts& opt,
               const std::function<bool(std::string_view)>& on_text);

}
