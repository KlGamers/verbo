#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "verbo/verbo.hpp"

namespace cli {

struct Args {
  std::string cmd;
  std::map<std::string, std::string> kv;
  std::vector<std::string> pos;

  bool has(const std::string& k) const { return kv.count(k) > 0; }
  std::string str(const std::string& k, const std::string& d = "") const {
    const auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  int num(const std::string& k, int d) const { return has(k) ? std::atoi(kv.at(k).c_str()) : d; }
  float flt(const std::string& k, float d) const { return has(k) ? std::strtof(kv.at(k).c_str(), nullptr) : d; }
};

inline const std::map<std::string, std::string> kAlias = {{"-m", "--model"},   {"-p", "--prompt"}, {"-n", "--tokens"},
                                                   {"-t", "--threads"}, {"-q", "--quant"},  {"-o", "--out"},
                                                   {"-s", "--system"},  {"-i", "--input"}};
inline const char* kFlags[] = {"--raw", "--stats", "--quiet"};

inline Args parse(int argc, char** argv) {
  Args a;
  if (argc > 1) a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    const auto al = kAlias.find(k);
    if (al != kAlias.end()) k = al->second;
    bool flag = false;
    for (const char* f : kFlags) flag = flag || k == f;
    if (flag) {
      a.kv[k] = "1";
    } else if (k.rfind("--", 0) == 0 && i + 1 < argc) {
      a.kv[k] = argv[++i];
    } else {
      a.pos.push_back(argv[i]);
    }
  }
  return a;
}

inline std::shared_ptr<const verbo::Model> load(const Args& a) {
  if (!a.has("--model")) throw std::runtime_error("missing -m MODEL");
  const auto t0 = std::chrono::steady_clock::now();
  auto m = verbo::Model::load(a.str("--model"), verbo::parse_dtype(a.str("--quant", "q8")));
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (!a.has("--quiet")) {
    const auto& c = m->cfg();
    std::fprintf(stderr, "[%s %d layers, dim %d, %s, %.0f MB, loaded in %.0f ms]\n", c.arch.c_str(), c.layers, c.dim,
                 verbo::dtype_name(m->dtype()), static_cast<double>(m->weight_bytes()) / 1e6, ms);
  }
  return m;
}

inline verbo::GenOpts opts(const Args& a, const verbo::Model& m) {
  verbo::GenOpts o;
  o.max_tokens = a.num("--tokens", 256);
  o.smp = m.sampling();
  o.smp.temp = a.flt("--temp", o.smp.temp);
  o.smp.top_k = a.num("--top-k", o.smp.top_k);
  o.smp.top_p = a.flt("--top-p", o.smp.top_p);
  o.smp.min_p = a.flt("--min-p", o.smp.min_p);
  o.smp.repeat = a.flt("--repeat", o.smp.repeat);
  o.smp.seed = static_cast<std::uint64_t>(std::strtoull(a.str("--seed", "0").c_str(), nullptr, 10));
  return o;
}

inline void report(const verbo::Stats& st) {
  std::fprintf(stderr, "\n[prompt %d tok %.1f tok/s | gen %d tok %.1f tok/s]\n", st.prompt,
               st.prompt_ms > 0 ? st.prompt * 1000.0 / st.prompt_ms : 0.0, st.gen,
               st.gen_ms > 0 ? st.gen * 1000.0 / st.gen_ms : 0.0);
}

int cmd_serve(const Args& a);

}
