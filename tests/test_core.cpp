#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "attn.hpp"
#include "json.hpp"
#include "kernels.hpp"
#include "pool.hpp"
#include "unicode.hpp"
#include "verbo/verbo.h"
#include "verbo/verbo.hpp"

namespace fs = std::filesystem;
using namespace verbo;

namespace {

int g_fail = 0;

#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++g_fail;                                                     \
    }                                                               \
  } while (0)

template <typename F>
bool throws(F&& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

std::string slurp(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

const fs::path kData = "tests/data";
const char* kModels[] = {"tiny-llama", "tiny-qwen2"};

std::vector<float> ref_logits(const fs::path& dir) {
  const std::string raw = slurp(dir / "ref_logits.bin");
  std::vector<float> out(raw.size() / 4);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

std::vector<int> ref_ids(const Json& r) {
  std::vector<int> ids;
  for (const Json& v : r["ids"].arr()) ids.push_back(static_cast<int>(v.as_int()));
  return ids;
}

double kl(const float* a, const float* b, int n) {
  const float ma = *std::max_element(a, a + n), mb = *std::max_element(b, b + n);
  double sa = 0, sb = 0;
  for (int i = 0; i < n; ++i) {
    sa += std::exp(static_cast<double>(a[i] - ma));
    sb += std::exp(static_cast<double>(b[i] - mb));
  }
  double k = 0;
  for (int i = 0; i < n; ++i) {
    const double pa = std::exp(static_cast<double>(a[i] - ma)) / sa;
    const double pb = std::exp(static_cast<double>(b[i] - mb)) / sb;
    if (pa > 0) k += pa * (std::log(pa) - std::log(pb + 1e-300));
  }
  return k;
}

void test_json() {
  const Json j = Json::parse(R"({"a": [1, 2.5, -3e2, true, false, null], "s": "x\"\\\n\u00e9\ud83d\ude00", "o": {}})");
  CHECK(j["a"].size() == 6);
  CHECK(j["a"].arr()[2].as_num() == -300.0);
  CHECK(j["a"].arr()[3].as_bool());
  CHECK(j["a"].arr()[5].is_null());
  CHECK(j["s"].as_str() == "x\"\\\n\xC3\xA9\xF0\x9F\x98\x80");
  CHECK(j["o"].is_obj() && j["o"].size() == 0);
  CHECK(j["missing"].is_null());
  CHECK(j.num_or("missing", 7.0) == 7.0);
  CHECK(Json::parse(j.dump()).dump() == j.dump());
  CHECK(Json::parse("[NaN]").arr()[0].is_num());
  CHECK(throws([] { Json::parse("{\"a\": }"); }));
  CHECK(throws([] { Json::parse("[1, 2"); }));
  CHECK(throws([] { Json::parse("\"abc"); }));
  CHECK(throws([] { Json::parse("{} x"); }));
  CHECK(throws([] { Json::parse("\"\\q\""); }));
  CHECK(throws([] { Json::parse("[1]").as_str(); }));
  std::string deep(600, '[');
  CHECK(throws([&] { Json::parse(deep); }));
}

void test_unicode() {
  CHECK(uni::nfc("e\xCC\x81") == "\xC3\xA9");
  CHECK(uni::nfc("A\xCC\x8A") == "\xC3\x85");
  CHECK(uni::nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8") == "\xEA\xB0\x81");
  CHECK(uni::nfc("a\xCC\xA3\xCC\x87") == "\xE1\xBA\xA1\xCC\x87");
  CHECK(uni::nfc("a\xCC\x87\xCC\xA3") == "\xE1\xBA\xA1\xCC\x87");
  CHECK(uni::nfc("\xEF\xAC\x81") == "\xEF\xAC\x81");
  CHECK(uni::nfc("plain ascii") == "plain ascii");
  CHECK(uni::nfc("\xFF\xC3\xA9") == "\xFF\xC3\xA9");
  const auto cps = uni::decode("a\xC3\xA9\xFF\xF0\x9F\x98\x80");
  CHECK(cps.size() == 4);
  CHECK(cps[1].cp == 0xE9 && cps[2].cp == uni::kBad && cps[3].cp == 0x1F600);
  CHECK(uni::valid_prefix("ab\xC3") == 2);
  CHECK(uni::valid_prefix("ab\xF0\x9F\x98") == 2);
  CHECK(uni::valid_prefix("ab\xF0\x9F\x98\x80") == 6);
  CHECK(uni::is_letter(0xF1) && uni::is_letter(0x6771) && !uni::is_letter('1'));
  CHECK(uni::is_number(0x0663) && uni::is_number(0xBD) && !uni::is_number('a'));
  CHECK(uni::is_space(0x3000) && uni::is_space(0xA0) && !uni::is_space(0x200B));
}

void test_tokenizer() {
  for (const char* name : kModels) {
    const fs::path dir = kData / name;
    const auto tok = Tokenizer::parse(slurp(dir / "tokenizer.json"));
    const Json r = Json::parse(slurp(dir / "ref.json"));
    int ok = 0, total = 0;
    for (const Json& c : r["cases"].arr()) {
      std::vector<int> want;
      for (const Json& v : c["ids"].arr()) want.push_back(static_cast<int>(v.as_int()));
      const auto got = tok.encode(c["text"].as_str());
      ++total;
      if (got == want) {
        ++ok;
      } else {
        std::printf("  mismatch in %s for \"%s\"\n", name, c["text"].as_str().c_str());
      }
      const std::string txt = c["text"].as_str();
      if (std::string(name) == "tiny-llama") CHECK(tok.decode(got) == txt);
    }
    CHECK(ok == total);
    CHECK(tok.encode(tok.decode(ref_ids(r))) == ref_ids(r));
    const int ims = tok.find("<|im_start|>");
    CHECK(ims >= 0 && tok.is_special(ims));
    const auto lit = tok.encode("<|im_start|>", false);
    CHECK(lit.size() > 1);
    CHECK(tok.decode(tok.encode("a<|im_end|>b"), true) == "ab");
    CHECK(tok.encode("").empty());
    CHECK(tok.find("no-such-token-xyz") == -1);
    CHECK(tok.piece(-1).empty() && tok.piece(1 << 30).empty());
  }
  CHECK(throws([] { Tokenizer::parse(R"({"model": {"type": "Unigram"}})"); }));
  CHECK(throws([] { Tokenizer::parse("not json"); }));
}

void test_detok() {
  const auto tok = Tokenizer::parse(slurp(kData / "tiny-llama" / "tokenizer.json"));
  const std::string txt = "café 😀 東京";
  std::string out;
  Detok dt(tok);
  for (const int id : tok.encode(txt)) {
    const std::string piece = dt.push(id);
    CHECK(uni::valid_prefix(piece) == piece.size());
    out += piece;
  }
  out += dt.flush();
  CHECK(out == txt);
}

void test_fp16() {
  const float vals[] = {0.0f, 1.0f, -2.5f, 65504.0f, 6.1035156e-05f, 5.9604645e-08f, 0.333333f, -1e-3f};
  for (const float v : vals) {
    const float r = h2f(f2h(v));
    CHECK(std::fabs(r - v) <= std::fabs(v) * 1e-3f + 1e-7f);
  }
  CHECK(std::isinf(h2f(f2h(1e6f))));
}

void test_quant() {
  std::mt19937 rng(3);
  std::normal_distribution<float> nd;
  const int rows = 13, cols = 96;
  std::vector<float> w(static_cast<std::size_t>(rows) * cols);
  for (auto& v : w) v = nd(rng);
  w[5] = 0.0f;
  std::fill(w.begin() + cols, w.begin() + 2 * cols, 0.0f);
  for (const DType dt : {DType::F32, DType::Q8, DType::Q4}) {
    Tensor t;
    t.dt = dt;
    t.rows = rows;
    t.cols = cols;
    std::vector<std::uint8_t> q(t.qbytes());
    std::vector<std::uint16_t> s(t.sbytes() / 2 + 1);
    quant_rows(dt, w.data(), rows, cols, q.data(), s.data());
    t.q = q.data();
    t.s = dt == DType::F32 ? nullptr : s.data();
    double err = 0, mag = 0;
    std::vector<float> row(static_cast<std::size_t>(cols));
    for (int r = 0; r < rows; ++r) {
      deq_row(t, r, row.data());
      for (int c = 0; c < cols; ++c) {
        const double d = row[static_cast<std::size_t>(c)] - w[static_cast<std::size_t>(r) * cols + c];
        err += d * d;
        mag += static_cast<double>(w[static_cast<std::size_t>(r) * cols + c]) * w[static_cast<std::size_t>(r) * cols + c];
      }
    }
    const double rel = std::sqrt(err / mag);
    CHECK(rel < (dt == DType::F32 ? 1e-9 : dt == DType::Q8 ? 0.01 : 0.12));
    deq_row(t, 1, row.data());
    CHECK(*std::max_element(row.begin(), row.end()) == 0.0f);
  }
  std::vector<float> x(64), xd(64);
  for (auto& v : x) v = nd(rng);
  std::vector<std::int8_t> xq(64);
  float xs[2];
  quant_act(x.data(), 64, xq.data(), xs);
  for (int i = 0; i < 64; ++i) CHECK(std::fabs(xq[static_cast<std::size_t>(i)] * xs[i / 32] - x[static_cast<std::size_t>(i)]) <= xs[i / 32] * 0.51f);
  std::vector<float> z(32, 0.0f);
  quant_act(z.data(), 32, xq.data(), xs);
  CHECK(xs[0] == 0.0f && xq[0] == 0);
}

void test_matmul() {
  std::mt19937 rng(5);
  std::normal_distribution<float> nd;
  for (const int threads : {1, 3, 4}) {
    Pool pool(threads);
    for (const DType dt : {DType::F32, DType::Q8, DType::Q4}) {
      for (const int rows : {4, 7, 130}) {
        for (const int n : {1, 2, 3, 9, 17}) {
          const int cols = 160;
          std::vector<float> w(static_cast<std::size_t>(rows) * cols), x(static_cast<std::size_t>(n) * cols);
          for (auto& v : w) v = nd(rng);
          for (auto& v : x) v = nd(rng);
          Tensor t;
          t.dt = dt;
          t.rows = rows;
          t.cols = cols;
          std::vector<std::uint8_t> q(t.qbytes());
          std::vector<std::uint16_t> s(t.sbytes() / 2 + 1);
          quant_rows(dt, w.data(), rows, cols, q.data(), s.data());
          t.q = q.data();
          t.s = dt == DType::F32 ? nullptr : s.data();
          std::vector<std::int8_t> xq(x.size());
          std::vector<float> xs(x.size() / kBlk);
          for (int i = 0; i < n; ++i) {
            quant_act(x.data() + static_cast<std::size_t>(i) * cols, cols, xq.data() + static_cast<std::size_t>(i) * cols,
                      xs.data() + static_cast<std::size_t>(i) * (cols / kBlk));
          }
          Act a;
          a.f = x.data();
          a.q = xq.data();
          a.s = xs.data();
          a.n = n;
          a.ld = cols;
          const int ldy = rows + 3;
          std::vector<float> y(static_cast<std::size_t>(n) * ldy, -7.0f);
          matmul(t, a, y.data(), ldy, pool);
          double worst = 0;
          std::vector<float> wr(static_cast<std::size_t>(cols));
          for (int r = 0; r < rows; ++r) {
            deq_row(t, r, wr.data());
            for (int i = 0; i < n; ++i) {
              double ref = 0, mag = 0;
              for (int c = 0; c < cols; ++c) {
                const double xv = dt == DType::F32
                                      ? x[static_cast<std::size_t>(i) * cols + c]
                                      : xq[static_cast<std::size_t>(i) * cols + c] * xs[static_cast<std::size_t>(i) * (cols / kBlk) + c / kBlk];
                ref += wr[static_cast<std::size_t>(c)] * xv;
                mag += std::fabs(wr[static_cast<std::size_t>(c)] * xv);
              }
              worst = std::max(worst, std::fabs(ref - y[static_cast<std::size_t>(i) * ldy + r]) / (mag + 1e-6));
            }
          }
          CHECK(worst < 1e-4);
          for (int i = 0; i < n; ++i) {
            for (int r = rows; r < ldy; ++r) CHECK(y[static_cast<std::size_t>(i) * ldy + r] == -7.0f);
          }
        }
      }
    }
  }
}

void test_attention() {
  std::mt19937 rng(9);
  std::normal_distribution<float> nd;
  struct Case {
    int heads, kvh, hd, n, p0, threads;
  };
  const Case cases[] = {{4, 2, 64, 1, 0, 1},   {4, 2, 64, 1, 300, 4}, {4, 4, 64, 5, 70, 3},
                        {6, 2, 64, 17, 0, 4},  {2, 1, 128, 3, 130, 2}, {4, 2, 32, 9, 64, 4},
                        {8, 1, 64, 1, 1000, 4}};
  for (const Case& cs : cases) {
    Pool pool(cs.threads);
    const int total = cs.p0 + cs.n;
    const int kvd = cs.kvh * cs.hd, qd = cs.heads * cs.hd;
    const int nblk = (total + kKvBlk - 1) / kKvBlk;
    const std::size_t kth = static_cast<std::size_t>(nblk) * cs.hd * kKvBlk;
    std::vector<float> k(static_cast<std::size_t>(total) * kvd), v(k.size()), q(static_cast<std::size_t>(cs.n) * qd);
    for (auto& x : k) x = nd(rng);
    for (auto& x : v) x = nd(rng);
    for (auto& x : q) x = nd(rng);
    std::vector<float> kt(kth * cs.kvh, 0.0f);
    for (int p = 0; p < total; ++p) {
      for (int g = 0; g < cs.kvh; ++g) {
        for (int d = 0; d < cs.hd; ++d) {
          kt[g * kth + static_cast<std::size_t>(p / kKvBlk) * cs.hd * kKvBlk + static_cast<std::size_t>(d) * kKvBlk + p % kKvBlk] =
              k[static_cast<std::size_t>(p) * kvd + g * cs.hd + d];
        }
      }
    }
    std::vector<float> out(static_cast<std::size_t>(cs.n) * qd);
    const float scale = 1.0f / std::sqrt(static_cast<float>(cs.hd));
    AttnArgs aa{q.data(), qd, kt.data(), kth, v.data(), kvd, out.data(), qd, cs.n, cs.p0, cs.heads, cs.kvh, cs.hd, scale};
    AttnScratch sc;
    attention(aa, pool, sc);
    double worst = 0;
    for (int i = 0; i < cs.n; ++i) {
      for (int h = 0; h < cs.heads; ++h) {
        const int g = h / (cs.heads / cs.kvh);
        const int len = cs.p0 + i + 1;
        std::vector<double> sc2(static_cast<std::size_t>(len));
        double mx = -1e300;
        for (int t = 0; t < len; ++t) {
          double s = 0;
          for (int d = 0; d < cs.hd; ++d) {
            s += static_cast<double>(q[static_cast<std::size_t>(i) * qd + h * cs.hd + d]) * k[static_cast<std::size_t>(t) * kvd + g * cs.hd + d];
          }
          sc2[static_cast<std::size_t>(t)] = s * scale;
          mx = std::max(mx, sc2[static_cast<std::size_t>(t)]);
        }
        double sum = 0;
        for (auto& s : sc2) {
          s = std::exp(s - mx);
          sum += s;
        }
        for (int d = 0; d < cs.hd; ++d) {
          double o = 0;
          for (int t = 0; t < len; ++t) o += sc2[static_cast<std::size_t>(t)] * v[static_cast<std::size_t>(t) * kvd + g * cs.hd + d];
          worst = std::max(worst, std::fabs(o / sum - out[static_cast<std::size_t>(i) * qd + h * cs.hd + d]));
        }
      }
    }
    CHECK(worst < 2e-5);
  }
  std::vector<float> e = {-100.0f, -10.0f, -1.0f, 0.0f, 0.5f, 1.0f, 10.0f, 80.0f, -0.3f};
  std::vector<float> e2 = e;
  exp_inplace(e2.data(), static_cast<int>(e2.size()));
  for (std::size_t i = 0; i < e.size(); ++i) CHECK(std::fabs(e2[i] - std::exp(e[i])) <= std::exp(e[i]) * 3e-7f + 1e-38f);
  std::vector<float> g = {-3, -1, 0, 1, 2, 5, -8, 0.5f, 7}, u = {1, 2, 3, 4, 5, 6, 7, 8, 9}, o(9);
  silu_mul(g.data(), u.data(), o.data(), 9);
  for (int i = 0; i < 9; ++i) CHECK(std::fabs(o[i] - g[i] / (1 + std::exp(-g[i])) * u[i]) < 1e-5f);
}

void test_pool() {
  Pool pool(4);
  for (int it = 0; it < 2000; ++it) {
    const int n = 1 + it % 37;
    std::vector<std::atomic<int>> hits(static_cast<std::size_t>(n));
    pool.run(n, [&](int t, int tid) {
      CHECK(tid >= 0 && tid < pool.size());
      hits[static_cast<std::size_t>(t)].fetch_add(1);
    });
    for (auto& h : hits) CHECK(h.load() == 1);
    if (it % 500 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  pool.run(0, [&](int, int) { CHECK(false); });
  CHECK(Pool::default_threads() >= 1);
}

void test_model() {
  for (const char* name : kModels) {
    const fs::path dir = kData / name;
    const Json r = Json::parse(slurp(dir / "ref.json"));
    const auto ids = ref_ids(r);
    const auto ref = ref_logits(dir);
    const int nv = static_cast<int>(r["vocab"].as_int());
    for (const DType dt : {DType::F32, DType::Q8, DType::Q4}) {
      const auto m = Model::load(dir.string(), dt);
      CHECK(m->cfg().vocab == nv);
      CHECK(m->dtype() == dt);
      Session ses(m, 0, 3);
      const auto all = ses.eval_all(ids);
      CHECK(ses.pos() == static_cast<int>(ids.size()));
      double worst = 0, kmean = 0;
      int agree = 0;
      for (std::size_t t = 0; t < ids.size(); ++t) {
        const float* a = ref.data() + t * nv;
        const float* b = all.data() + t * nv;
        for (int i = 0; i < nv; ++i) worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
        kmean += kl(a, b, nv);
        agree += std::max_element(a, a + nv) - a == std::max_element(b, b + nv) - b;
      }
      kmean /= static_cast<double>(ids.size());
      if (dt == DType::F32) {
        CHECK(worst < 1e-3);
        CHECK(agree == static_cast<int>(ids.size()));
      } else {
        CHECK(kmean < (dt == DType::Q8 ? 0.01 : 0.15));
      }
      Session inc(m, 0, 2);
      const float* last = nullptr;
      for (std::size_t t = 0; t < ids.size(); t += 5) {
        const int k = std::min<int>(5, static_cast<int>(ids.size() - t));
        last = inc.eval(ids.data() + t, k);
      }
      double d = 0;
      for (int i = 0; i < nv; ++i) d = std::max(d, static_cast<double>(std::fabs(last[i] - all[(ids.size() - 1) * nv + i])));
      CHECK(d < 1e-4);
      inc.rewind(3);
      const float* re = inc.eval(ids.data() + 3, static_cast<int>(ids.size()) - 3);
      d = 0;
      for (int i = 0; i < nv; ++i) d = std::max(d, static_cast<double>(std::fabs(re[i] - all[(ids.size() - 1) * nv + i])));
      CHECK(d < 1e-4);
      CHECK(throws([&] { inc.rewind(1000); }));
      CHECK(throws([&] { inc.eval(nullptr, 0); }));
      const int bad = 1 << 20;
      CHECK(throws([&] { Session(m, 0, 1).eval(&bad, 1); }));
      Session small(m, 8, 1);
      CHECK(throws([&] { small.eval(ids); }));
      const fs::path vrb = fs::temp_directory_path() / (std::string(name) + "-" + dtype_name(dt) + ".vrb");
      m->save(vrb.string());
      const auto m2 = Model::load(vrb.string());
      CHECK(m2->dtype() == dt);
      CHECK(m2->stops() == m->stops());
      CHECK(m2->weight_bytes() == m->weight_bytes());
      CHECK(m2->chat({{"user", "hola"}}) == m->chat({{"user", "hola"}}));
      Session s2(m2, 0, 3);
      const auto all2 = s2.eval_all(ids);
      CHECK(all2 == all);
      fs::remove(vrb);
    }
    const auto m = Model::load(dir.string(), DType::Q8);
    const std::string chat = m->chat({{"user", "hola"}});
    CHECK(chat == "<|im_start|>system\nEres un asistente diminuto.<|im_end|>\n<|im_start|>user\nhola<|im_end|>\n<|im_start|>assistant\n");
    CHECK(m->chat({{"system", "S"}, {"user", "u"}}, false) == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nu<|im_end|>\n");
    CHECK(m->is_stop(2) && m->is_stop(0) && !m->is_stop(5));
    const Sampling smp = m->sampling();
    CHECK(std::fabs(smp.temp - 0.6f) < 1e-6f && smp.top_k == 30 && std::fabs(smp.top_p - 0.85f) < 1e-6f);
  }
  CHECK(throws([] { Model::load("tests/data/does-not-exist"); }));
  CHECK(throws([] { Model::load("tests/data/tiny-llama/config.json"); }));
}

void test_sampler() {
  std::vector<float> lg = {0.1f, 3.0f, 2.9f, -1.0f, 0.0f};
  Sampling g;
  g.temp = 0.0f;
  CHECK(Sampler(g).pick(lg.data(), 5) == 1);
  Sampling s;
  s.seed = 42;
  s.top_k = 2;
  s.top_p = 1.0f;
  Sampler a(s), b(s);
  int c1 = 0;
  for (int i = 0; i < 400; ++i) {
    const int x = a.pick(lg.data(), 5);
    CHECK(x == b.pick(lg.data(), 5));
    CHECK(x == 1 || x == 2);
    c1 += x == 1;
  }
  CHECK(c1 > 150 && c1 < 300);
  Sampling p;
  p.seed = 1;
  p.top_k = 0;
  p.top_p = 0.5f;
  Sampler sp(p);
  for (int i = 0; i < 100; ++i) CHECK(sp.pick(lg.data(), 5) == 1);
  Sampling mp;
  mp.seed = 1;
  mp.top_k = 0;
  mp.top_p = 1.0f;
  mp.min_p = 0.5f;
  Sampler sm(mp);
  for (int i = 0; i < 100; ++i) {
    const int x = sm.pick(lg.data(), 5);
    CHECK(x == 1 || x == 2);
  }
  Sampling rp;
  rp.temp = 0.0f;
  rp.repeat = 2.0f;
  Sampler sr(rp);
  sr.accept(1);
  CHECK(sr.pick(lg.data(), 5) == 2);
  sr.reset();
  CHECK(sr.pick(lg.data(), 5) == 1);
}

void test_generate() {
  const auto m = Model::load((kData / "tiny-qwen2").string(), DType::Q8);
  Session ses(m, 0, 2);
  const auto ids = m->tok().encode(m->chat({{"user", "hola"}}));
  GenOpts o;
  o.max_tokens = 12;
  o.smp.temp = 0.0f;
  std::string a;
  const Stats st = generate(ses, ids, o, [&](std::string_view t) {
    a += t;
    return true;
  });
  CHECK(st.prompt == static_cast<int>(ids.size()));
  CHECK(st.gen <= 12);
  CHECK(st.gen == 12 || st.stopped);
  ses.rewind(0);
  std::string b;
  generate(ses, ids, o, [&](std::string_view t) {
    b += t;
    return true;
  });
  CHECK(a == b);
  if (a.size() > 2) {
    ses.rewind(0);
    o.stop_text = {a.substr(1, 2)};
    std::string c;
    const Stats s2 = generate(ses, ids, o, [&](std::string_view t) {
      c += t;
      return true;
    });
    CHECK(c == a.substr(0, a.find(o.stop_text[0])));
    CHECK(s2.stopped);
  }
  ses.rewind(0);
  int calls = 0;
  o.stop_text.clear();
  generate(ses, ids, o, [&](std::string_view) { return ++calls < 2; });
  CHECK(calls <= 2);
}

int text_cb(const char* t, size_t len, void* user) {
  static_cast<std::string*>(user)->append(t, len);
  return 1;
}

void test_capi() {
  verbo_model* m = verbo_model_load("tests/data/tiny-llama", "q8");
  CHECK(m != nullptr);
  CHECK(verbo_model_load("tests/data/nope", "q8") == nullptr);
  CHECK(std::string(verbo_last_error()).find("cannot open") != std::string::npos);
  CHECK(verbo_model_load("tests/data/tiny-llama", "q3") == nullptr);
  CHECK(verbo_model_vocab(m) > 0);
  const char* txt = "hola mundo";
  const int n = verbo_tokenize(m, txt, std::strlen(txt), 1, nullptr, 0);
  CHECK(n > 0);
  std::vector<int> ids(static_cast<std::size_t>(n));
  CHECK(verbo_tokenize(m, txt, std::strlen(txt), 1, ids.data(), n) == n);
  char buf[64];
  CHECK(verbo_detokenize(m, ids.data(), n, 0, buf, sizeof buf) == static_cast<int>(std::strlen(txt)));
  CHECK(std::string(buf) == txt);
  CHECK(verbo_detokenize(m, ids.data(), n, 0, buf, 4) == static_cast<int>(std::strlen(txt)) && std::string(buf) == "hol");
  const char* roles[] = {"user"};
  const char* texts[] = {"hola"};
  const int need = verbo_chat_prompt(m, roles, texts, 1, 1, nullptr, 0);
  std::string prompt(static_cast<std::size_t>(need) + 1, '\0');
  verbo_chat_prompt(m, roles, texts, 1, 1, prompt.data(), need + 1);
  prompt.resize(static_cast<std::size_t>(need));
  CHECK(prompt.find("<|im_start|>assistant\n") != std::string::npos);
  verbo_session* s = verbo_session_new(m, 64, 2);
  CHECK(s != nullptr && verbo_session_ctx(s) == 64);
  CHECK(verbo_session_eval(s, ids.data(), n) != nullptr);
  CHECK(verbo_session_pos(s) == n);
  CHECK(verbo_session_rewind(s, 0) == 0);
  CHECK(verbo_session_rewind(s, 99) == -1);
  verbo_gen_opts o = verbo_model_defaults(m);
  o.max_tokens = 5;
  o.temp = 0.0f;
  std::string out;
  CHECK(verbo_generate(s, ids.data(), n, &o, text_cb, &out) >= 0);
  const int big = 1 << 22;
  CHECK(verbo_session_eval(s, &big, 1) == nullptr);
  const fs::path vrb = fs::temp_directory_path() / "capi.vrb";
  CHECK(verbo_model_save(m, vrb.string().c_str()) == 0);
  verbo_model* m2 = verbo_model_load(vrb.string().c_str(), nullptr);
  CHECK(m2 != nullptr);
  verbo_model_free(m2);
  fs::remove(vrb);
  CHECK(verbo_model_is_stop(m, 2) == 1);
  CHECK(std::string(verbo_version()).size() > 0);
  verbo_session_free(s);
  verbo_model_free(m);
}

}

int main() {
  struct T {
    const char* name;
    std::function<void()> fn;
  };
  const std::vector<T> all = {{"json", test_json},           {"unicode", test_unicode}, {"tokenizer", test_tokenizer},
                              {"detok", test_detok},         {"fp16", test_fp16},       {"quant", test_quant},
                              {"matmul", test_matmul},       {"attention", test_attention}, {"pool", test_pool},
                              {"model", test_model},         {"sampler", test_sampler}, {"generate", test_generate},
                              {"capi", test_capi}};
  std::printf("simd: %s\n", simd_name());
  for (const T& t : all) {
    const int before = g_fail;
    std::printf("%s\n", t.name);
    t.fn();
    std::printf("  %s\n", g_fail == before ? "ok" : "FAILED");
  }
  std::printf("%d failure(s)\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
