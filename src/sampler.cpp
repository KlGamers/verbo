#include <algorithm>
#include <chrono>
#include <cmath>

#include "verbo/verbo.hpp"

namespace verbo {

namespace {

std::uint64_t splitmix(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}

Sampler::Sampler(Sampling cfg) : cfg_(cfg) { reset(); }

void Sampler::reset() {
  std::uint64_t s = cfg_.seed ? cfg_.seed : static_cast<std::uint64_t>(now_ms() * 1000.0);
  rng_[0] = splitmix(s);
  rng_[1] = splitmix(s);
  hist_.clear();
}

double Sampler::unit() {
  std::uint64_t s1 = rng_[0];
  const std::uint64_t s0 = rng_[1];
  const std::uint64_t r = s0 + s1;
  rng_[0] = s0;
  s1 ^= s1 << 23;
  rng_[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
  return static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
}

void Sampler::accept(int tok) {
  hist_.push_back(tok);
  if (cfg_.repeat_last > 0 && hist_.size() > static_cast<std::size_t>(cfg_.repeat_last) * 4) {
    hist_.erase(hist_.begin(), hist_.end() - cfg_.repeat_last);
  }
}

int Sampler::pick(const float* logits, int n) {
  buf_.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) buf_[static_cast<std::size_t>(i)] = {logits[i], i};
  if (cfg_.repeat != 1.0f && !hist_.empty()) {
    const std::size_t from =
        cfg_.repeat_last > 0 && hist_.size() > static_cast<std::size_t>(cfg_.repeat_last) ? hist_.size() - cfg_.repeat_last
                                                                                          : 0;
    std::vector<int> seen(hist_.begin() + static_cast<std::ptrdiff_t>(from), hist_.end());
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    for (const int t : seen) {
      if (t < 0 || t >= n) continue;
      float& v = buf_[static_cast<std::size_t>(t)].first;
      v = v > 0.0f ? v / cfg_.repeat : v * cfg_.repeat;
    }
  }
  if (cfg_.temp <= 0.0f) {
    return std::max_element(buf_.begin(), buf_.end(),
                            [](const auto& a, const auto& b) { return a.first < b.first || (a.first == b.first && a.second > b.second); })
        ->second;
  }
  auto by = [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
  std::size_t k = buf_.size();
  if (cfg_.top_k > 0 && static_cast<std::size_t>(cfg_.top_k) < k) {
    k = static_cast<std::size_t>(cfg_.top_k);
    std::partial_sort(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(k), buf_.end(), by);
  } else {
    std::sort(buf_.begin(), buf_.end(), by);
  }
  const float mx = buf_[0].first;
  double sum = 0.0;
  std::vector<double> pr(k);
  for (std::size_t i = 0; i < k; ++i) {
    pr[i] = std::exp(static_cast<double>(buf_[i].first - mx) / cfg_.temp);
    sum += pr[i];
  }
  std::size_t keep = k;
  if (cfg_.min_p > 0.0f) {
    const double thr = pr[0] * cfg_.min_p;
    keep = 1;
    while (keep < k && pr[keep] >= thr) ++keep;
  }
  if (cfg_.top_p > 0.0f && cfg_.top_p < 1.0f) {
    double acc = 0.0;
    std::size_t i = 0;
    for (; i < keep; ++i) {
      acc += pr[i] / sum;
      if (acc >= cfg_.top_p) break;
    }
    keep = std::min(keep, i + 1);
  }
  double tot = 0.0;
  for (std::size_t i = 0; i < keep; ++i) tot += pr[i];
  double r = unit() * tot;
  for (std::size_t i = 0; i < keep; ++i) {
    r -= pr[i];
    if (r <= 0.0) return buf_[i].second;
  }
  return buf_[keep - 1].second;
}

Stats generate(Session& ses, const std::vector<int>& prompt, const GenOpts& opt,
               const std::function<bool(std::string_view)>& on_text) {
  Stats st;
  const Model& m = ses.model();
  const int nv = m.cfg().vocab;
  Sampler smp(opt.smp);
  for (const int t : prompt) smp.accept(t);
  st.prompt = static_cast<int>(prompt.size());
  double t0 = now_ms();
  const float* lg = ses.eval(prompt);
  st.prompt_ms = now_ms() - t0;
  Detok dt(m.tok());
  std::string pend;
  std::size_t hold = 0;
  for (const auto& s : opt.stop_text) hold = std::max(hold, s.size());
  auto emit = [&](bool fin) {
    for (const auto& s : opt.stop_text) {
      const std::size_t at = s.empty() ? std::string::npos : pend.find(s);
      if (at != std::string::npos) {
        if (at) on_text(std::string_view(pend).substr(0, at));
        pend.clear();
        return false;
      }
    }
    std::size_t safe = pend.size();
    if (!fin && hold) {
      for (std::size_t k = std::min(pend.size(), hold - 1); k > 0; --k) {
        bool pre = false;
        for (const auto& s : opt.stop_text) {
          if (s.size() > k && pend.compare(pend.size() - k, k, s, 0, k) == 0) pre = true;
        }
        if (pre) {
          safe = pend.size() - k;
          break;
        }
      }
    }
    if (safe) {
      const bool go = on_text(std::string_view(pend).substr(0, safe));
      pend.erase(0, safe);
      return go;
    }
    return true;
  };
  t0 = now_ms();
  bool go = true;
  while (go && st.gen < opt.max_tokens) {
    const int id = smp.pick(lg, nv);
    if (m.is_stop(id)) {
      st.stopped = true;
      break;
    }
    smp.accept(id);
    ++st.gen;
    st.toks.push_back(id);
    pend += dt.push(id);
    go = emit(false);
    if (!go) {
      st.stopped = true;
      break;
    }
    if (ses.pos() >= ses.ctx() || st.gen >= opt.max_tokens) break;
    lg = ses.eval(&id, 1);
  }
  if (go) {
    pend += dt.flush();
    emit(true);
  }
  st.gen_ms = now_ms() - t0;
  return st;
}

}
