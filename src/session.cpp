#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "attn.hpp"
#include "kernels.hpp"
#include "model.hpp"

namespace verbo {

namespace {

void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  const float r = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) out[i] = w[i] * (x[i] * r);
}

}

struct Session::Impl {
  std::shared_ptr<const Model> mdl;
  const Model::Impl* m;
  Config c;
  int ctx;
  int pos = 0;
  int bmax = 64;
  int qd, kvd, qkvd, in_max, nblk;
  std::size_t kth;
  Pool pool;
  std::vector<float> kc, vc, cosv, sinv;
  std::vector<float> x, xb, tmp, qkv, att, gu, h, logits;
  std::vector<std::int8_t> xq;
  std::vector<float> xs;
  AttnScratch asc;

  Impl(std::shared_ptr<const Model> md, int cx, int th)
      : mdl(std::move(md)), m(&mdl->impl()), c(m->cfg), ctx(cx > 0 ? cx : std::min(c.max_ctx, 4096)),
        pool(th > 0 ? th : Pool::default_threads()) {
    qd = c.heads * c.head_dim;
    kvd = c.kv_heads * c.head_dim;
    qkvd = qd + 2 * kvd;
    in_max = std::max({c.dim, qd, c.hidden});
    nblk = (ctx + kKvBlk - 1) / kKvBlk;
    kth = static_cast<std::size_t>(nblk) * c.head_dim * kKvBlk;
    kc.assign(static_cast<std::size_t>(c.layers) * c.kv_heads * kth, 0.0f);
    vc.assign(static_cast<std::size_t>(c.layers) * ctx * kvd, 0.0f);
    const int half = c.head_dim / 2;
    std::vector<float> inv(static_cast<std::size_t>(half));
    for (int i = 0; i < half; ++i) {
      float f = 1.0f / std::pow(c.theta, static_cast<float>(2 * i) / static_cast<float>(c.head_dim));
      if (c.rope_factor > 0.0f) {
        const float lo_wl = static_cast<float>(c.rope_orig) / c.rope_low;
        const float hi_wl = static_cast<float>(c.rope_orig) / c.rope_high;
        const float wl = 2.0f * static_cast<float>(M_PI) / f;
        if (wl > lo_wl) {
          f /= c.rope_factor;
        } else if (wl >= hi_wl) {
          const float sm = (static_cast<float>(c.rope_orig) / wl - c.rope_low) / (c.rope_high - c.rope_low);
          f = (1.0f - sm) * f / c.rope_factor + sm * f;
        }
      }
      inv[static_cast<std::size_t>(i)] = f;
    }
    cosv.resize(static_cast<std::size_t>(ctx) * half);
    sinv.resize(cosv.size());
    for (int p = 0; p < ctx; ++p) {
      for (int i = 0; i < half; ++i) {
        const float a = static_cast<float>(p) * inv[static_cast<std::size_t>(i)];
        cosv[static_cast<std::size_t>(p) * half + i] = std::cos(a);
        sinv[static_cast<std::size_t>(p) * half + i] = std::sin(a);
      }
    }
    const auto b = static_cast<std::size_t>(bmax);
    x.resize(b * c.dim);
    xb.resize(b * c.dim);
    tmp.resize(b * c.dim);
    qkv.resize(b * qkvd);
    att.resize(b * qd);
    gu.resize(b * 2 * c.hidden);
    h.resize(b * c.hidden);
    xq.resize(b * in_max);
    xs.resize(b * in_max / kBlk);
  }

  Act act(const Tensor& w, const float* src, int n, int cols) {
    Act a;
    a.f = src;
    a.n = n;
    a.ld = cols;
    if (w.dt != DType::F32) {
      for (int i = 0; i < n; ++i) {
        quant_act(src + static_cast<std::size_t>(i) * cols, cols, xq.data() + static_cast<std::size_t>(i) * cols,
                  xs.data() + static_cast<std::size_t>(i) * (cols / kBlk));
      }
      a.q = xq.data();
      a.s = xs.data();
    }
    return a;
  }

  void rope(float* v, int p) {
    const int half = c.head_dim / 2;
    const float* cs = cosv.data() + static_cast<std::size_t>(p) * half;
    const float* sn = sinv.data() + static_cast<std::size_t>(p) * half;
    for (int i = 0; i < half; ++i) {
      const float a = v[i], b = v[i + half];
      v[i] = a * cs[i] - b * sn[i];
      v[i + half] = a * sn[i] + b * cs[i];
    }
  }

  void batch(const int* toks, int n, float* out, bool all) {
    const int dim = c.dim, hd = c.head_dim, hid = c.hidden;
    const int p0 = pos;
    for (int i = 0; i < n; ++i) {
      if (toks[i] < 0 || toks[i] >= m->emb.rows) throw std::runtime_error("verbo: token id out of range");
      deq_row(m->emb, toks[i], x.data() + static_cast<std::size_t>(i) * dim);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int l = 0; l < c.layers; ++l) {
      const Layer& ly = m->lay[static_cast<std::size_t>(l)];
      float* kl = kc.data() + static_cast<std::size_t>(l) * c.kv_heads * kth;
      float* vl = vc.data() + static_cast<std::size_t>(l) * ctx * kvd;
      for (int i = 0; i < n; ++i) {
        rmsnorm(x.data() + static_cast<std::size_t>(i) * dim, ly.an, xb.data() + static_cast<std::size_t>(i) * dim, dim,
                c.eps);
      }
      matmul(ly.qkv, act(ly.qkv, xb.data(), n, dim), qkv.data(), qkvd, pool);
      for (int i = 0; i < n; ++i) {
        float* r = qkv.data() + static_cast<std::size_t>(i) * qkvd;
        if (ly.qkv_b) {
          for (int j = 0; j < qkvd; ++j) r[j] += ly.qkv_b[j];
        }
        for (int hh = 0; hh < c.heads + c.kv_heads; ++hh) rope(r + hh * hd, p0 + i);
        const int ps = p0 + i;
        for (int g = 0; g < c.kv_heads; ++g) {
          float* kb = kl + g * kth + static_cast<std::size_t>(ps / kKvBlk) * hd * kKvBlk + ps % kKvBlk;
          const float* kr = r + qd + g * hd;
          for (int d = 0; d < hd; ++d) kb[static_cast<std::size_t>(d) * kKvBlk] = kr[d];
        }
        std::memcpy(vl + static_cast<std::size_t>(p0 + i) * kvd, r + qd + kvd, sizeof(float) * kvd);
      }
      AttnArgs aa{qkv.data(), qkvd, kl, kth, vl, kvd, att.data(), qd, n, p0, c.heads, c.kv_heads, hd, scale};
      attention(aa, pool, asc);
      matmul(ly.o, act(ly.o, att.data(), n, qd), tmp.data(), dim, pool);
      for (std::size_t j = 0; j < static_cast<std::size_t>(n) * dim; ++j) x[j] += tmp[j];
      for (int i = 0; i < n; ++i) {
        rmsnorm(x.data() + static_cast<std::size_t>(i) * dim, ly.fn, xb.data() + static_cast<std::size_t>(i) * dim, dim,
                c.eps);
      }
      matmul(ly.gu, act(ly.gu, xb.data(), n, dim), gu.data(), 2 * hid, pool);
      for (int i = 0; i < n; ++i) {
        const float* g = gu.data() + static_cast<std::size_t>(i) * 2 * hid;
        silu_mul(g, g + hid, h.data() + static_cast<std::size_t>(i) * hid, hid);
      }
      matmul(ly.down, act(ly.down, h.data(), n, hid), tmp.data(), dim, pool);
      for (std::size_t j = 0; j < static_cast<std::size_t>(n) * dim; ++j) x[j] += tmp[j];
    }
    pos += n;
    const int first = all ? 0 : n - 1;
    const int cnt = n - first;
    for (int i = first; i < n; ++i) {
      rmsnorm(x.data() + static_cast<std::size_t>(i) * dim, m->norm,
              xb.data() + static_cast<std::size_t>(i - first) * dim, dim, c.eps);
    }
    matmul(m->head, act(m->head, xb.data(), cnt, dim), out, c.vocab, pool);
  }

  void run(const int* toks, int n, float* out, bool all) {
    if (n <= 0) throw std::runtime_error("verbo: empty input");
    if (pos + n > ctx) throw std::runtime_error("verbo: context window exceeded");
    for (int st = 0; st < n; st += bmax) {
      const int k = std::min(bmax, n - st);
      const bool last = st + k == n;
      float* dst = all ? out + static_cast<std::size_t>(st) * c.vocab : out;
      if (all || last) {
        batch(toks + st, k, dst, all);
      } else {
        batch(toks + st, k, logits.data(), false);
      }
    }
  }
};

Session::Session(std::shared_ptr<const Model> mdl, int ctx, int threads)
    : impl_(std::make_unique<Impl>(std::move(mdl), ctx, threads)) {
  impl_->logits.resize(static_cast<std::size_t>(impl_->c.vocab));
}

Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

const float* Session::eval(const int* toks, int n) {
  impl_->run(toks, n, impl_->logits.data(), false);
  return impl_->logits.data();
}

std::vector<float> Session::eval_all(const std::vector<int>& toks) {
  std::vector<float> out(toks.size() * static_cast<std::size_t>(impl_->c.vocab));
  impl_->run(toks.data(), static_cast<int>(toks.size()), out.data(), true);
  return out;
}

int Session::pos() const { return impl_->pos; }
int Session::ctx() const { return impl_->ctx; }
int Session::threads() const { return impl_->pool.size(); }
const Model& Session::model() const { return *impl_->mdl; }

void Session::rewind(int p) {
  if (p < 0 || p > impl_->pos) throw std::runtime_error("verbo: invalid rewind position");
  impl_->pos = p;
}

}
