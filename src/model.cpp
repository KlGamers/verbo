#include "model.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "json.hpp"

namespace verbo {

namespace fs = std::filesystem;

namespace {

constexpr char kMagic[4] = {'V', 'R', 'B', 'O'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kAlign = 64;

std::string slurp(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("verbo: cannot open " + p.string());
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct Src {
  const std::uint8_t* base = nullptr;
  std::string dt;
  std::vector<std::int64_t> shape;
};

void to_f32(const Src& s, std::size_t at, std::size_t n, float* out) {
  if (s.dt == "F32") {
    std::memcpy(out, s.base + at * 4, n * 4);
  } else if (s.dt == "BF16") {
    const auto* p = reinterpret_cast<const std::uint16_t*>(s.base) + at;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t b = static_cast<std::uint32_t>(p[i]) << 16;
      std::memcpy(out + i, &b, 4);
    }
  } else if (s.dt == "F16") {
    const auto* p = reinterpret_cast<const std::uint16_t*>(s.base) + at;
    for (std::size_t i = 0; i < n; ++i) out[i] = h2f(p[i]);
  } else {
    throw std::runtime_error("verbo: unsupported tensor dtype " + s.dt);
  }
}

std::vector<int> ids_of(const Json& v) {
  std::vector<int> out;
  if (v.is_num()) out.push_back(static_cast<int>(v.as_int()));
  if (v.is_arr()) {
    for (const Json& x : v.arr()) {
      if (x.is_num()) out.push_back(static_cast<int>(x.as_int()));
    }
  }
  return out;
}

Config parse_config(const Json& j) {
  Config c;
  c.arch = j.str_or("model_type", "");
  if (c.arch != "llama" && c.arch != "qwen2" && c.arch != "mistral") {
    throw std::runtime_error("verbo: unsupported architecture '" + c.arch + "'");
  }
  c.dim = static_cast<int>(j["hidden_size"].as_int());
  c.hidden = static_cast<int>(j["intermediate_size"].as_int());
  c.layers = static_cast<int>(j["num_hidden_layers"].as_int());
  c.heads = static_cast<int>(j["num_attention_heads"].as_int());
  c.kv_heads = static_cast<int>(j.num_or("num_key_value_heads", c.heads));
  c.head_dim = static_cast<int>(j.num_or("head_dim", c.dim / c.heads));
  c.vocab = static_cast<int>(j["vocab_size"].as_int());
  c.max_ctx = static_cast<int>(j.num_or("max_position_embeddings", 4096));
  c.eps = static_cast<float>(j.num_or("rms_norm_eps", 1e-5));
  const Json& rp = j["rope_parameters"];
  c.theta = static_cast<float>(j.num_or("rope_theta", rp.num_or("rope_theta", 10000.0)));
  c.tied = j.bool_or("tie_word_embeddings", false);
  c.qkv_bias = c.arch == "qwen2" || j.bool_or("attention_bias", false);
  const Json& rs = j["rope_scaling"].is_obj() ? j["rope_scaling"] : rp;
  if (rs.is_obj()) {
    const std::string ty = rs.str_or("rope_type", rs.str_or("type", "default"));
    if (ty == "llama3") {
      c.rope_factor = static_cast<float>(rs.num_or("factor", 8.0));
      c.rope_low = static_cast<float>(rs.num_or("low_freq_factor", 1.0));
      c.rope_high = static_cast<float>(rs.num_or("high_freq_factor", 4.0));
      c.rope_orig = static_cast<int>(rs.num_or("original_max_position_embeddings", 8192));
    } else if (ty != "default") {
      throw std::runtime_error("verbo: unsupported rope scaling " + ty);
    }
  }
  if (j.bool_or("use_sliding_window", false)) throw std::runtime_error("verbo: sliding window attention is not supported");
  if (j["layer_types"].is_arr()) {
    for (const Json& t : j["layer_types"].arr()) {
      if (t.is_str() && t.as_str() != "full_attention") throw std::runtime_error("verbo: only full attention layers are supported");
    }
  }
  if (c.dim % kBlk || c.hidden % kBlk || c.head_dim % 2) throw std::runtime_error("verbo: unsupported dimensions");
  if (c.heads % c.kv_heads) throw std::runtime_error("verbo: heads must be a multiple of kv heads");
  return c;
}

Json config_json(const Config& c) {
  std::string s = "{";
  s += "\"model_type\":";
  json_escape(s, c.arch);
  s += ",\"hidden_size\":" + std::to_string(c.dim);
  s += ",\"intermediate_size\":" + std::to_string(c.hidden);
  s += ",\"num_hidden_layers\":" + std::to_string(c.layers);
  s += ",\"num_attention_heads\":" + std::to_string(c.heads);
  s += ",\"num_key_value_heads\":" + std::to_string(c.kv_heads);
  s += ",\"head_dim\":" + std::to_string(c.head_dim);
  s += ",\"vocab_size\":" + std::to_string(c.vocab);
  s += ",\"max_position_embeddings\":" + std::to_string(c.max_ctx);
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.9g", c.eps);
  s += std::string(",\"rms_norm_eps\":") + buf;
  std::snprintf(buf, sizeof buf, "%.9g", c.theta);
  s += std::string(",\"rope_theta\":") + buf;
  s += std::string(",\"tie_word_embeddings\":") + (c.tied ? "true" : "false");
  s += std::string(",\"attention_bias\":") + (c.qkv_bias ? "true" : "false");
  if (c.rope_factor > 0.0f) {
    std::snprintf(buf, sizeof buf, "%.9g", c.rope_factor);
    s += std::string(",\"rope_scaling\":{\"rope_type\":\"llama3\",\"factor\":") + buf;
    std::snprintf(buf, sizeof buf, "%.9g", c.rope_low);
    s += std::string(",\"low_freq_factor\":") + buf;
    std::snprintf(buf, sizeof buf, "%.9g", c.rope_high);
    s += std::string(",\"high_freq_factor\":") + buf;
    s += ",\"original_max_position_embeddings\":" + std::to_string(c.rope_orig) + "}";
  }
  s += "}";
  return Json::parse(s);
}

std::string find_system(const std::string& t) {
  const std::string key = "<|im_start|>system";
  std::size_t p = 0;
  while ((p = t.find(key, p)) != std::string::npos) {
    std::size_t b = p + key.size();
    if (t.compare(b, 1, "\n") == 0) {
      b += 1;
    } else if (t.compare(b, 2, "\\n") == 0) {
      b += 2;
    } else {
      p = b;
      continue;
    }
    const std::size_t e = t.find("<|im_end|>", b);
    if (e == std::string::npos) break;
    const std::string body = t.substr(b, e - b);
    if (!body.empty() && body.find_first_of("{}'\"+") == std::string::npos) return body;
    p = b;
  }
  return {};
}

void read_meta(Model::Impl& im, const fs::path& dir) {
  const Json tc = fs::exists(dir / "tokenizer_config.json") ? Json::parse(slurp(dir / "tokenizer_config.json")) : Json();
  const Json gc = fs::exists(dir / "generation_config.json") ? Json::parse(slurp(dir / "generation_config.json")) : Json();
  const Json cj = Json::parse(slurp(dir / "config.json"));
  const std::string tmpl = tc.str_or("chat_template", "");
  if (tmpl.find("<|im_start|>") != std::string::npos) {
    im.chat_kind = "chatml";
    im.sys = find_system(tmpl);
  }
  std::vector<int> st = ids_of(gc["eos_token_id"]);
  for (const int id : ids_of(cj["eos_token_id"])) st.push_back(id);
  if (tc["eos_token"].is_str()) st.push_back(im.tok.find(tc["eos_token"].as_str()));
  if (im.chat_kind == "chatml") st.push_back(im.tok.find("<|im_end|>"));
  for (const int id : st) {
    if (id >= 0 && std::find(im.stops.begin(), im.stops.end(), id) == im.stops.end()) im.stops.push_back(id);
  }
  if (gc.is_obj()) {
    im.dflt.temp = static_cast<float>(gc.num_or("temperature", im.dflt.temp));
    im.dflt.top_p = static_cast<float>(gc.num_or("top_p", im.dflt.top_p));
    im.dflt.top_k = static_cast<int>(gc.num_or("top_k", im.dflt.top_k));
    im.dflt.repeat = static_cast<float>(gc.num_or("repetition_penalty", im.dflt.repeat));
  }
}

std::shared_ptr<Model::Impl> load_hf(const fs::path& dir, DType dt) {
  auto im = std::make_shared<Model::Impl>();
  im->cfg = parse_config(Json::parse(slurp(dir / "config.json")));
  im->tok = Tokenizer::parse(slurp(dir / "tokenizer.json"));
  im->dt = dt;
  read_meta(*im, dir);
  const Config& c = im->cfg;

  std::vector<fs::path> files;
  if (fs::exists(dir / "model.safetensors.index.json")) {
    const Json idx = Json::parse(slurp(dir / "model.safetensors.index.json"));
    for (const auto& kv : idx["weight_map"].obj()) {
      const fs::path f = dir / kv.second.as_str();
      if (std::find(files.begin(), files.end(), f) == files.end()) files.push_back(f);
    }
  } else {
    files.push_back(dir / "model.safetensors");
  }
  std::map<std::string, Src> src;
  for (const auto& f : files) {
    auto mp = std::make_shared<Mapping>(f.string());
    if (mp->len < 8) throw std::runtime_error("verbo: truncated safetensors file");
    std::uint64_t hl = 0;
    std::memcpy(&hl, mp->data(), 8);
    if (8 + hl > mp->len) throw std::runtime_error("verbo: truncated safetensors header");
    const Json h = Json::parse(std::string_view(reinterpret_cast<const char*>(mp->data()) + 8, hl));
    const std::uint8_t* body = mp->data() + 8 + hl;
    for (const auto& kv : h.obj()) {
      if (kv.first == "__metadata__") continue;
      Src s;
      s.dt = kv.second["dtype"].as_str();
      for (const Json& d : kv.second["shape"].arr()) s.shape.push_back(d.as_int());
      const auto& off = kv.second["data_offsets"].arr();
      const auto a = static_cast<std::size_t>(off.at(0).as_int()), b = static_cast<std::size_t>(off.at(1).as_int());
      if (8 + hl + b > mp->len || a > b) throw std::runtime_error("verbo: tensor out of bounds: " + kv.first);
      s.base = body + a;
      src[kv.first] = s;
    }
    im->maps.push_back(mp);
  }
  auto get = [&](const std::string& n) -> const Src& {
    const auto it = src.find(n);
    if (it == src.end()) throw std::runtime_error("verbo: missing tensor " + n);
    return it->second;
  };

  Pool pool(Pool::default_threads());
  auto mat = [&](const std::vector<std::string>& names, DType tdt) {
    std::vector<const Src*> parts;
    std::vector<int> starts;
    int rows = 0, cols = -1;
    for (const auto& n : names) {
      const Src& s = get(n);
      if (s.shape.size() != 2) throw std::runtime_error("verbo: expected a matrix: " + n);
      if (cols >= 0 && s.shape[1] != cols) throw std::runtime_error("verbo: mismatched columns: " + n);
      cols = static_cast<int>(s.shape[1]);
      parts.push_back(&s);
      starts.push_back(rows);
      rows += static_cast<int>(s.shape[0]);
    }
    Tensor t;
    t.dt = tdt;
    t.rows = rows;
    t.cols = cols;
    auto* q = im->alloc(t.qbytes());
    std::uint16_t* sc = tdt == DType::F32 ? nullptr : reinterpret_cast<std::uint16_t*>(im->alloc(t.sbytes()));
    const int chunk = 64;
    std::vector<std::vector<float>> tmp(static_cast<std::size_t>(pool.size()));
    pool.run((rows + chunk - 1) / chunk, [&](int k, int tid) {
      auto& buf = tmp[static_cast<std::size_t>(tid)];
      buf.resize(static_cast<std::size_t>(chunk) * cols);
      const int r0 = k * chunk, r1 = std::min(rows, r0 + chunk);
      for (int r = r0; r < r1; ++r) {
        std::size_t p = parts.size() - 1;
        while (starts[p] > r) --p;
        to_f32(*parts[p], static_cast<std::size_t>(r - starts[p]) * cols, static_cast<std::size_t>(cols),
               buf.data() + static_cast<std::size_t>(r - r0) * cols);
      }
      void* dq = q + t.row_bytes() * static_cast<std::size_t>(r0);
      std::uint16_t* ds = sc ? sc + static_cast<std::size_t>(r0) * (cols / kBlk) : nullptr;
      quant_rows(tdt, buf.data(), r1 - r0, cols, dq, ds);
    });
    t.q = q;
    t.s = sc;
    return t;
  };
  auto vec = [&](const std::vector<std::string>& names) {
    std::size_t n = 0;
    for (const auto& nm : names) n += static_cast<std::size_t>(get(nm).shape.at(0));
    auto* out = reinterpret_cast<float*>(im->alloc(n * 4));
    std::size_t at = 0;
    for (const auto& nm : names) {
      const Src& s = get(nm);
      to_f32(s, 0, static_cast<std::size_t>(s.shape[0]), out + at);
      at += static_cast<std::size_t>(s.shape[0]);
    }
    return static_cast<const float*>(out);
  };

  const DType hdt = dt == DType::Q4 ? DType::Q8 : dt;
  im->emb = mat({"model.embed_tokens.weight"}, hdt);
  if (c.tied || !src.count("lm_head.weight")) {
    im->head = im->emb;
  } else {
    im->head = mat({"lm_head.weight"}, hdt);
  }
  im->norm = vec({"model.norm.weight"});
  for (int l = 0; l < c.layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    Layer ly;
    ly.an = vec({p + "input_layernorm.weight"});
    ly.fn = vec({p + "post_attention_layernorm.weight"});
    ly.qkv = mat({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"}, hdt);
    if (c.qkv_bias) ly.qkv_b = vec({p + "self_attn.q_proj.bias", p + "self_attn.k_proj.bias", p + "self_attn.v_proj.bias"});
    ly.o = mat({p + "self_attn.o_proj.weight"}, dt);
    ly.gu = mat({p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}, dt);
    ly.down = mat({p + "mlp.down_proj.weight"}, hdt);
    if (ly.qkv.rows != (c.heads + 2 * c.kv_heads) * c.head_dim || ly.qkv.cols != c.dim || ly.gu.rows != 2 * c.hidden) {
      throw std::runtime_error("verbo: tensor shapes do not match config");
    }
    im->lay.push_back(ly);
  }
  im->cfg.vocab = im->head.rows;
  im->maps.clear();
  return im;
}

struct Writer {
  std::ofstream out;
  std::uint64_t at = 0;

  void raw(const void* p, std::size_t n) {
    out.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    at += n;
  }

  void pad() {
    static const char z[kAlign] = {};
    const std::size_t r = at % kAlign;
    if (r) raw(z, kAlign - r);
  }
};

std::shared_ptr<Model::Impl> load_vrb(const std::string& path) {
  auto mp = std::make_shared<Mapping>(path);
  const std::uint8_t* d = mp->data();
  if (mp->len < 16 || std::memcmp(d, kMagic, 4) != 0) throw std::runtime_error("verbo: not a .vrb file");
  std::uint32_t ver = 0;
  std::uint64_t hl = 0;
  std::memcpy(&ver, d + 4, 4);
  std::memcpy(&hl, d + 8, 8);
  if (ver != kVersion) throw std::runtime_error("verbo: unsupported .vrb version");
  if (16 + hl > mp->len) throw std::runtime_error("verbo: truncated .vrb header");
  const Json h = Json::parse(std::string_view(reinterpret_cast<const char*>(d) + 16, hl));
  auto blob = [&](const Json& r) {
    const auto off = static_cast<std::size_t>(r.arr().at(0).as_int());
    const auto len = static_cast<std::size_t>(r.arr().at(1).as_int());
    if (off + len > mp->len || off % kAlign) throw std::runtime_error("verbo: corrupt .vrb blob");
    return std::make_pair(d + off, len);
  };
  auto im = std::make_shared<Model::Impl>();
  im->cfg = parse_config(h["config"]);
  im->cfg.vocab = static_cast<int>(h["config"]["vocab_size"].as_int());
  im->dt = parse_dtype(h["dtype"].as_str());
  const auto tk = blob(h["tokenizer"]);
  im->tok = Tokenizer::parse(std::string_view(reinterpret_cast<const char*>(tk.first), tk.second));
  im->chat_kind = h.str_or("chat", "");
  im->sys = h.str_or("system", "");
  im->bos = static_cast<int>(h.num_or("bos", -1));
  im->stops = ids_of(h["stops"]);
  const Json& df = h["defaults"];
  im->dflt.temp = static_cast<float>(df.num_or("temp", im->dflt.temp));
  im->dflt.top_p = static_cast<float>(df.num_or("top_p", im->dflt.top_p));
  im->dflt.top_k = static_cast<int>(df.num_or("top_k", im->dflt.top_k));
  im->dflt.repeat = static_cast<float>(df.num_or("repeat", im->dflt.repeat));
  std::map<std::string, Tensor> tens;
  for (const Json& t : h["tensors"].arr()) {
    Tensor x;
    x.dt = parse_dtype(t["dt"].as_str());
    x.rows = static_cast<int>(t["rows"].as_int());
    x.cols = static_cast<int>(t["cols"].as_int());
    const auto q = blob(t["q"]);
    if (q.second != x.qbytes()) throw std::runtime_error("verbo: corrupt .vrb tensor");
    x.q = q.first;
    if (x.dt != DType::F32) {
      const auto s = blob(t["s"]);
      if (s.second != x.sbytes()) throw std::runtime_error("verbo: corrupt .vrb tensor");
      x.s = reinterpret_cast<const std::uint16_t*>(s.first);
    }
    tens[t["name"].as_str()] = x;
  }
  auto ten = [&](const std::string& n) {
    const auto it = tens.find(n);
    if (it == tens.end()) throw std::runtime_error("verbo: .vrb is missing " + n);
    return it->second;
  };
  auto fv = [&](const std::string& n) { return static_cast<const float*>(ten(n).q); };
  const Config& c = im->cfg;
  im->emb = ten("emb");
  im->head = tens.count("head") ? ten("head") : im->emb;
  im->norm = fv("norm");
  for (int l = 0; l < c.layers; ++l) {
    const std::string p = "l" + std::to_string(l) + ".";
    Layer ly;
    ly.an = fv(p + "an");
    ly.fn = fv(p + "fn");
    if (c.qkv_bias) ly.qkv_b = fv(p + "qkv_b");
    ly.qkv = ten(p + "qkv");
    ly.o = ten(p + "o");
    ly.gu = ten(p + "gu");
    ly.down = ten(p + "down");
    im->lay.push_back(ly);
  }
  im->maps.push_back(mp);
  return im;
}

}

Mapping::Mapping(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("verbo: cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("verbo: cannot stat " + path);
  }
  len = static_cast<std::size_t>(st.st_size);
  if (len) {
    ptr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
      ptr = nullptr;
      ::close(fd);
      throw std::runtime_error("verbo: cannot map " + path);
    }
  }
  ::close(fd);
}

Mapping::~Mapping() {
  if (ptr) ::munmap(ptr, len);
}

Buf::Buf(std::size_t n) {
  const std::size_t sz = std::max<std::size_t>(kAlign, (n + kAlign - 1) / kAlign * kAlign);
  p = static_cast<std::uint8_t*>(std::aligned_alloc(kAlign, sz));
  if (!p) throw std::bad_alloc();
}

Buf::~Buf() { std::free(p); }

std::uint8_t* Model::Impl::alloc(std::size_t n) {
  own.push_back(std::make_unique<Buf>(n));
  return own.back()->p;
}

const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::F32: return "f32";
    case DType::Q8: return "q8";
    case DType::Q4: return "q4";
  }
  return "?";
}

DType parse_dtype(std::string_view s) {
  if (s == "f32") return DType::F32;
  if (s == "q8") return DType::Q8;
  if (s == "q4") return DType::Q4;
  throw std::runtime_error("verbo: unknown dtype '" + std::string(s) + "'");
}

std::shared_ptr<const Model> Model::load(const std::string& path, DType dt) {
  auto m = std::shared_ptr<Model>(new Model());
  if (fs::is_directory(path)) {
    m->impl_ = load_hf(path, dt);
  } else {
    m->impl_ = load_vrb(path);
  }
  return m;
}

void Model::save(const std::string& path) const {
  const Impl& im = *impl_;
  struct Item {
    std::string name;
    Tensor t;
  };
  std::vector<Item> items;
  auto vecten = [](const float* p, int n) {
    Tensor t;
    t.dt = DType::F32;
    t.rows = 1;
    t.cols = n;
    t.q = p;
    return t;
  };
  const Config& c = im.cfg;
  items.push_back({"emb", im.emb});
  if (im.head.q != im.emb.q) items.push_back({"head", im.head});
  items.push_back({"norm", vecten(im.norm, c.dim)});
  for (int l = 0; l < c.layers; ++l) {
    const std::string p = "l" + std::to_string(l) + ".";
    const Layer& ly = im.lay[static_cast<std::size_t>(l)];
    items.push_back({p + "an", vecten(ly.an, c.dim)});
    items.push_back({p + "fn", vecten(ly.fn, c.dim)});
    if (ly.qkv_b) items.push_back({p + "qkv_b", vecten(ly.qkv_b, ly.qkv.rows)});
    items.push_back({p + "qkv", ly.qkv});
    items.push_back({p + "o", ly.o});
    items.push_back({p + "gu", ly.gu});
    items.push_back({p + "down", ly.down});
  }
  const std::string& tj = im.tok.source();
  auto make_header = [&](std::uint64_t base) {
    std::uint64_t at = base;
    auto take = [&](std::size_t n) {
      const std::uint64_t o = at;
      at = (at + n + kAlign - 1) / kAlign * kAlign;
      return "[" + std::to_string(o) + "," + std::to_string(n) + "]";
    };
    std::string h = "{\"config\":" + config_json(c).dump();
    h += ",\"dtype\":\"" + std::string(dtype_name(im.dt)) + "\"";
    h += ",\"chat\":";
    json_escape(h, im.chat_kind);
    h += ",\"system\":";
    json_escape(h, im.sys);
    h += ",\"bos\":" + std::to_string(im.bos) + ",\"stops\":[";
    for (std::size_t i = 0; i < im.stops.size(); ++i) h += (i ? "," : "") + std::to_string(im.stops[i]);
    char buf[160];
    std::snprintf(buf, sizeof buf, "],\"defaults\":{\"temp\":%.6g,\"top_p\":%.6g,\"top_k\":%d,\"repeat\":%.6g}",
                  im.dflt.temp, im.dflt.top_p, im.dflt.top_k, im.dflt.repeat);
    h += buf;
    h += ",\"tokenizer\":" + take(tj.size()) + ",\"tensors\":[";
    for (std::size_t i = 0; i < items.size(); ++i) {
      const Tensor& t = items[i].t;
      if (i) h += ",";
      h += "{\"name\":";
      json_escape(h, items[i].name);
      h += ",\"dt\":\"" + std::string(dtype_name(t.dt)) + "\",\"rows\":" + std::to_string(t.rows) +
           ",\"cols\":" + std::to_string(t.cols) + ",\"q\":" + take(t.qbytes());
      if (t.dt != DType::F32) h += ",\"s\":" + take(t.sbytes());
      h += "}";
    }
    return h + "]}";
  };
  std::uint64_t base = kAlign;
  std::string h;
  for (;;) {
    h = make_header(base);
    const std::uint64_t need = (16 + h.size() + kAlign - 1) / kAlign * kAlign;
    if (need <= base) break;
    base = need;
  }
  h.resize(base - 16, ' ');
  Writer w;
  w.out.open(path, std::ios::binary | std::ios::trunc);
  if (!w.out) throw std::runtime_error("verbo: cannot write " + path);
  w.raw(kMagic, 4);
  w.raw(&kVersion, 4);
  const std::uint64_t hl = h.size();
  w.raw(&hl, 8);
  w.raw(h.data(), h.size());
  w.raw(tj.data(), tj.size());
  w.pad();
  for (const Item& it : items) {
    w.raw(it.t.q, it.t.qbytes());
    w.pad();
    if (it.t.dt != DType::F32) {
      w.raw(it.t.s, it.t.sbytes());
      w.pad();
    }
  }
  if (!w.out) throw std::runtime_error("verbo: write failed for " + path);
}

const Config& Model::cfg() const { return impl_->cfg; }
const Tokenizer& Model::tok() const { return impl_->tok; }
DType Model::dtype() const { return impl_->dt; }
int Model::bos() const { return impl_->bos; }
const std::vector<int>& Model::stops() const { return impl_->stops; }

bool Model::is_stop(int id) const {
  return std::find(impl_->stops.begin(), impl_->stops.end(), id) != impl_->stops.end();
}

std::size_t Model::weight_bytes() const {
  const Impl& im = *impl_;
  std::size_t n = im.emb.qbytes() + im.emb.sbytes();
  if (im.head.q != im.emb.q) n += im.head.qbytes() + im.head.sbytes();
  for (const Layer& l : im.lay) {
    for (const Tensor* t : {&l.qkv, &l.o, &l.gu, &l.down}) n += t->qbytes() + t->sbytes();
  }
  return n;
}

std::string Model::chat(const std::vector<Msg>& msgs, bool gen) const {
  const Impl& im = *impl_;
  if (im.chat_kind != "chatml") throw std::runtime_error("verbo: this model has no supported chat template");
  std::string out;
  if (!im.sys.empty() && (msgs.empty() || msgs[0].role != "system")) {
    out += "<|im_start|>system\n" + im.sys + "<|im_end|>\n";
  }
  for (const Msg& m : msgs) out += "<|im_start|>" + m.role + "\n" + m.text + "<|im_end|>\n";
  if (gen) out += "<|im_start|>assistant\n";
  return out;
}

Sampling Model::sampling() const {
  const Defaults& d = impl_->dflt;
  Sampling s;
  s.temp = d.temp;
  s.top_p = d.top_p;
  s.top_k = d.top_k;
  s.repeat = d.repeat;
  return s;
}

}
