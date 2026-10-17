#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

#if defined(VERBO_FORCE_SCALAR)
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define VERBO_NEON 1
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define VERBO_AVX2 1
#endif

#if !defined(VERBO_NEON) && defined(__F16C__)
#include <immintrin.h>
#endif

namespace verbo {

std::size_t Tensor::row_bytes() const {
  switch (dt) {
    case DType::F32: return static_cast<std::size_t>(cols) * 4;
    case DType::Q8: return static_cast<std::size_t>(cols);
    case DType::Q4: return static_cast<std::size_t>(cols) / 2;
  }
  return 0;
}

std::size_t Tensor::sbytes() const {
  if (dt == DType::F32) return 0;
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols / kBlk) * 2;
}

float h2f(std::uint16_t h) {
#if defined(VERBO_NEON)
  __fp16 v;
  std::memcpy(&v, &h, 2);
  return static_cast<float>(v);
#elif defined(__F16C__)
  return _cvtsh_ss(h);
#else
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
  std::uint32_t exp = (h >> 10) & 0x1F;
  std::uint32_t man = h & 0x3FF;
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FF;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000 | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
#endif
}

std::uint16_t f2h(float f) {
#if defined(VERBO_NEON)
  const __fp16 v = static_cast<__fp16>(f);
  std::uint16_t h;
  std::memcpy(&h, &v, 2);
  return h;
#elif defined(__F16C__)
  return _cvtss_sh(f, 0);
#else
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000;
  const std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
  std::uint32_t man = x & 0x7FFFFF;
  if (((x >> 23) & 0xFF) == 0xFF) return static_cast<std::uint16_t>(sign | 0x7C00 | (man ? 0x200 : 0));
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return static_cast<std::uint16_t>(sign);
    man |= 0x800000;
    const int sh = 14 - exp;
    std::uint32_t r = man >> sh;
    const std::uint32_t rem = man & ((1u << sh) - 1);
    const std::uint32_t half = 1u << (sh - 1);
    if (rem > half || (rem == half && (r & 1))) ++r;
    return static_cast<std::uint16_t>(sign | r);
  }
  std::uint32_t r = (static_cast<std::uint32_t>(exp) << 10) | (man >> 13);
  const std::uint32_t rem = man & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
  return static_cast<std::uint16_t>(sign | r);
#endif
}

void quant_act(const float* x, int n, std::int8_t* q, float* s) {
  for (int b = 0; b < n / kBlk; ++b) {
    const float* v = x + b * kBlk;
    float amax = 0.0f;
#if defined(VERBO_NEON)
    float32x4_t m = vdupq_n_f32(0.0f);
    for (int i = 0; i < kBlk; i += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(v + i)));
    amax = vmaxvq_f32(m);
#else
    for (int i = 0; i < kBlk; ++i) amax = std::max(amax, std::fabs(v[i]));
#endif
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    s[b] = d;
    std::int8_t* o = q + b * kBlk;
#if defined(VERBO_NEON)
    for (int i = 0; i < kBlk; i += 8) {
      const int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(v + i), id));
      const int32x4_t c = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(v + i + 4), id));
      const int16x8_t h = vcombine_s16(vmovn_s32(a), vmovn_s32(c));
      vst1_s8(o + i, vmovn_s16(h));
    }
#else
    for (int i = 0; i < kBlk; ++i) o[i] = static_cast<std::int8_t>(std::nearbyint(v[i] * id));
#endif
  }
}

void quant_rows(DType dt, const float* x, int rows, int cols, void* q, std::uint16_t* s) {
  const int nb = cols / kBlk;
  for (int r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    if (dt == DType::F32) {
      std::memcpy(static_cast<float*>(q) + static_cast<std::size_t>(r) * cols, row, sizeof(float) * cols);
      continue;
    }
    for (int b = 0; b < nb; ++b) {
      const float* v = row + b * kBlk;
      std::uint16_t* sd = s + static_cast<std::size_t>(r) * nb + b;
      if (dt == DType::Q8) {
        float amax = 0.0f;
        for (int i = 0; i < kBlk; ++i) amax = std::max(amax, std::fabs(v[i]));
        *sd = f2h(amax / 127.0f);
        const float d = h2f(*sd);
        const float id = d ? 1.0f / d : 0.0f;
        auto* o = static_cast<std::int8_t*>(q) + static_cast<std::size_t>(r) * cols + b * kBlk;
        for (int i = 0; i < kBlk; ++i) {
          o[i] = static_cast<std::int8_t>(std::clamp(static_cast<int>(std::nearbyint(v[i] * id)), -127, 127));
        }
      } else {
        float amax = 0.0f, mx = 0.0f;
        for (int i = 0; i < kBlk; ++i) {
          if (std::fabs(v[i]) > amax) {
            amax = std::fabs(v[i]);
            mx = v[i];
          }
        }
        *sd = f2h(mx / -8.0f);
        const float d = h2f(*sd);
        const float id = d ? 1.0f / d : 0.0f;
        auto* o = static_cast<std::uint8_t*>(q) + static_cast<std::size_t>(r) * (cols / 2) + b * (kBlk / 2);
        for (int i = 0; i < kBlk / 2; ++i) {
          const int lo = std::clamp(static_cast<int>(v[i] * id + 8.5f), 0, 15);
          const int hi = std::clamp(static_cast<int>(v[i + kBlk / 2] * id + 8.5f), 0, 15);
          o[i] = static_cast<std::uint8_t>(lo | (hi << 4));
        }
      }
    }
  }
}

void deq_row(const Tensor& t, int r, float* out) {
  const int nb = t.cols / kBlk;
  if (t.dt == DType::F32) {
    std::memcpy(out, static_cast<const float*>(t.q) + static_cast<std::size_t>(r) * t.cols, sizeof(float) * t.cols);
    return;
  }
  for (int b = 0; b < nb; ++b) {
    const float d = h2f(t.s[static_cast<std::size_t>(r) * nb + b]);
    float* o = out + b * kBlk;
    if (t.dt == DType::Q8) {
      const auto* w = static_cast<const std::int8_t*>(t.q) + static_cast<std::size_t>(r) * t.cols + b * kBlk;
      for (int i = 0; i < kBlk; ++i) o[i] = d * w[i];
    } else {
      const auto* w = static_cast<const std::uint8_t*>(t.q) + static_cast<std::size_t>(r) * (t.cols / 2) + b * 16;
      for (int i = 0; i < 16; ++i) {
        o[i] = d * static_cast<float>((w[i] & 0x0F) - 8);
        o[i + 16] = d * static_cast<float>((w[i] >> 4) - 8);
      }
    }
  }
}

#if defined(VERBO_NEON)

namespace {

inline int32x4_t sdot(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
  return vdotq_s32(acc, a, b);
#else
  const int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
  const int16x8_t hi = vmull_high_s8(a, b);
  return vpadalq_s16(vpadalq_s16(acc, lo), hi);
#endif
}

}

float dot_f32(const float* a, const float* b, int n) {
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float r = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < n; ++i) r += a[i] * b[i];
  return r;
}

float dot_q8(const std::int8_t* w, const std::uint16_t* ws, const std::int8_t* x, const float* xs, int nb) {
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0;
  int b = 0;
  const int32x4_t z = vdupq_n_s32(0);
  for (; b + 2 <= nb; b += 2) {
    const std::int8_t* p = w + b * kBlk;
    const std::int8_t* q = x + b * kBlk;
    const int32x4_t d0 = sdot(sdot(z, vld1q_s8(p), vld1q_s8(q)), vld1q_s8(p + 16), vld1q_s8(q + 16));
    const int32x4_t d1 = sdot(sdot(z, vld1q_s8(p + 32), vld1q_s8(q + 32)), vld1q_s8(p + 48), vld1q_s8(q + 48));
    a0 = vmlaq_n_f32(a0, vcvtq_f32_s32(d0), h2f(ws[b]) * xs[b]);
    a1 = vmlaq_n_f32(a1, vcvtq_f32_s32(d1), h2f(ws[b + 1]) * xs[b + 1]);
  }
  for (; b < nb; ++b) {
    const std::int8_t* p = w + b * kBlk;
    const std::int8_t* q = x + b * kBlk;
    const int32x4_t d0 = sdot(sdot(z, vld1q_s8(p), vld1q_s8(q)), vld1q_s8(p + 16), vld1q_s8(q + 16));
    a0 = vmlaq_n_f32(a0, vcvtq_f32_s32(d0), h2f(ws[b]) * xs[b]);
  }
  return vaddvq_f32(vaddq_f32(a0, a1));
}

float dot_q4(const std::uint8_t* w, const std::uint16_t* ws, const std::int8_t* x, const float* xs, int nb) {
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0;
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
  const int8x16_t s8 = vdupq_n_s8(8);
  const int32x4_t z = vdupq_n_s32(0);
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    const uint8x16_t v0 = vld1q_u8(w + b * 16);
    const uint8x16_t v1 = vld1q_u8(w + b * 16 + 16);
    const int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4)), s8);
    const int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8);
    const int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4)), s8);
    const int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), s8);
    const std::int8_t* q = x + b * kBlk;
    const int32x4_t d0 = sdot(sdot(z, l0, vld1q_s8(q)), h0, vld1q_s8(q + 16));
    const int32x4_t d1 = sdot(sdot(z, l1, vld1q_s8(q + 32)), h1, vld1q_s8(q + 48));
    a0 = vmlaq_n_f32(a0, vcvtq_f32_s32(d0), h2f(ws[b]) * xs[b]);
    a1 = vmlaq_n_f32(a1, vcvtq_f32_s32(d1), h2f(ws[b + 1]) * xs[b + 1]);
  }
  for (; b < nb; ++b) {
    const uint8x16_t v0 = vld1q_u8(w + b * 16);
    const int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4)), s8);
    const int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8);
    const std::int8_t* q = x + b * kBlk;
    const int32x4_t d0 = sdot(sdot(z, l0, vld1q_s8(q)), h0, vld1q_s8(q + 16));
    a0 = vmlaq_n_f32(a0, vcvtq_f32_s32(d0), h2f(ws[b]) * xs[b]);
  }
  return vaddvq_f32(vaddq_f32(a0, a1));
}


struct Pack {
  std::vector<std::int8_t> q;
  std::vector<float> s;
  int pairs = 0;
};

namespace {

template <typename Ld>
inline void rows4_one(Ld ld, const std::uint16_t* s0, const std::uint16_t* s1, const std::uint16_t* s2,
                      const std::uint16_t* s3, const std::int8_t* x, const float* xs, int nb, float* y) {
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
  const int32x4_t z = vdupq_n_s32(0);
  for (int b = 0; b < nb; ++b) {
    const int8x16_t x0 = vld1q_s8(x + b * kBlk), x1 = vld1q_s8(x + b * kBlk + 16);
    int8x16_t l, h;
    const float dx = xs[b];
    ld(0, b, l, h);
    a0 = vmlaq_n_f32(a0, vcvtq_f32_s32(sdot(sdot(z, l, x0), h, x1)), h2f(s0[b]) * dx);
    ld(1, b, l, h);
    a1 = vmlaq_n_f32(a1, vcvtq_f32_s32(sdot(sdot(z, l, x0), h, x1)), h2f(s1[b]) * dx);
    ld(2, b, l, h);
    a2 = vmlaq_n_f32(a2, vcvtq_f32_s32(sdot(sdot(z, l, x0), h, x1)), h2f(s2[b]) * dx);
    ld(3, b, l, h);
    a3 = vmlaq_n_f32(a3, vcvtq_f32_s32(sdot(sdot(z, l, x0), h, x1)), h2f(s3[b]) * dx);
  }
  vst1q_f32(y, vpaddq_f32(vpaddq_f32(a0, a1), vpaddq_f32(a2, a3)));
}

}

void rows1(const Tensor& w, const std::int8_t* x, const float* xs, int r0, int r1, float* y) {
  const int nb = w.cols / kBlk;
  const std::size_t rb = w.row_bytes();
  const auto* base = static_cast<const std::uint8_t*>(w.q);
  int r = r0;
  for (; r + 4 <= r1; r += 4) {
    const std::uint8_t* p[4] = {base + rb * r, base + rb * (r + 1), base + rb * (r + 2), base + rb * (r + 3)};
    const std::uint16_t* sc = w.s + static_cast<std::size_t>(r) * nb;
    if (w.dt == DType::Q8) {
      rows4_one(
          [&](int k, int b, int8x16_t& l, int8x16_t& h) {
            const auto* q = reinterpret_cast<const std::int8_t*>(p[k]) + b * kBlk;
            l = vld1q_s8(q);
            h = vld1q_s8(q + 16);
          },
          sc, sc + nb, sc + 2 * nb, sc + 3 * nb, x, xs, nb, y + r);
    } else {
      const uint8x16_t m4 = vdupq_n_u8(0x0F);
      const int8x16_t s8 = vdupq_n_s8(8);
      rows4_one(
          [&](int k, int b, int8x16_t& l, int8x16_t& h) {
            const uint8x16_t v = vld1q_u8(p[k] + b * 16);
            l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), s8);
            h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), s8);
          },
          sc, sc + nb, sc + 2 * nb, sc + 3 * nb, x, xs, nb, y + r);
    }
  }
  for (; r < r1; ++r) {
    const std::uint8_t* p = base + rb * r;
    const std::uint16_t* sc = w.s + static_cast<std::size_t>(r) * nb;
    y[r] = w.dt == DType::Q8 ? dot_q8(reinterpret_cast<const std::int8_t*>(p), sc, x, xs, nb) : dot_q4(p, sc, x, xs, nb);
  }
}

#if defined(__ARM_FEATURE_MATMUL_INT8)

bool has_mm() { return true; }

void pack_act(const Act& x, Pack& pk) {
  const int nb = x.ld / kBlk;
  pk.pairs = (x.n + 1) / 2;
  pk.q.resize(static_cast<std::size_t>(pk.pairs) * nb * 64);
  pk.s.resize(static_cast<std::size_t>(pk.pairs) * nb * 4);
  for (int p = 0; p < pk.pairs; ++p) {
    const int t0 = 2 * p, t1 = std::min(2 * p + 1, x.n - 1);
    const bool dup = t1 == t0;
    const std::int8_t* a = x.q + static_cast<std::size_t>(t0) * x.ld;
    const std::int8_t* c = x.q + static_cast<std::size_t>(t1) * x.ld;
    for (int b = 0; b < nb; ++b) {
      std::int8_t* o = pk.q.data() + (static_cast<std::size_t>(p) * nb + b) * 64;
      for (int k = 0; k < 4; ++k) {
        std::memcpy(o + 16 * k, a + b * kBlk + 8 * k, 8);
        std::memcpy(o + 16 * k + 8, c + b * kBlk + 8 * k, 8);
      }
      const float d0 = x.s[static_cast<std::size_t>(t0) * nb + b];
      const float d1 = dup ? 0.0f : x.s[static_cast<std::size_t>(t1) * nb + b];
      float* sv = pk.s.data() + (static_cast<std::size_t>(p) * nb + b) * 4;
      sv[0] = d0;
      sv[1] = d1;
      sv[2] = d0;
      sv[3] = d1;
    }
  }
}

namespace {

template <int TP>
inline void tile(const std::int8_t* wb, const float* dwb, const Pack& pk, int p0, int nb, float32x4_t* out) {
  float32x4_t f[2][TP];
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < TP; ++j) f[i][j] = vdupq_n_f32(0.0f);
  }
  const int32x4_t z = vdupq_n_s32(0);
  const std::int8_t* xq = pk.q.data() + static_cast<std::size_t>(p0) * nb * 64;
  const float* xs = pk.s.data() + static_cast<std::size_t>(p0) * nb * 4;
  const std::size_t xstr = static_cast<std::size_t>(nb) * 64;
  const std::size_t sstr = static_cast<std::size_t>(nb) * 4;
  for (int b = 0; b < nb; ++b) {
    const std::int8_t* wp = wb + static_cast<std::size_t>(b) * 128;
    const int8x16_t a0 = vld1q_s8(wp), a1 = vld1q_s8(wp + 16), a2 = vld1q_s8(wp + 32), a3 = vld1q_s8(wp + 48);
    const int8x16_t c0 = vld1q_s8(wp + 64), c1 = vld1q_s8(wp + 80), c2 = vld1q_s8(wp + 96), c3 = vld1q_s8(wp + 112);
    const float32x4_t d0 = vld1q_f32(dwb + b * 8), d1 = vld1q_f32(dwb + b * 8 + 4);
#pragma clang loop unroll(full)
    for (int j = 0; j < TP; ++j) {
      const std::int8_t* xp = xq + j * xstr + static_cast<std::size_t>(b) * 64;
      const int8x16_t x0 = vld1q_s8(xp), x1 = vld1q_s8(xp + 16), x2 = vld1q_s8(xp + 32), x3 = vld1q_s8(xp + 48);
      const float32x4_t dx = vld1q_f32(xs + j * sstr + b * 4);
      const int32x4_t u0 = vmmlaq_s32(vmmlaq_s32(z, a0, x0), a1, x1);
      const int32x4_t u1 = vmmlaq_s32(vmmlaq_s32(z, a2, x2), a3, x3);
      const int32x4_t v0 = vmmlaq_s32(vmmlaq_s32(z, c0, x0), c1, x1);
      const int32x4_t v1 = vmmlaq_s32(vmmlaq_s32(z, c2, x2), c3, x3);
      f[0][j] = vfmaq_f32(f[0][j], vcvtq_f32_s32(vaddq_s32(u0, u1)), vmulq_f32(d0, dx));
      f[1][j] = vfmaq_f32(f[1][j], vcvtq_f32_s32(vaddq_s32(v0, v1)), vmulq_f32(d1, dx));
    }
  }
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < TP; ++j) out[i * TP + j] = f[i][j];
  }
}

}

void mm_rows(const Tensor& w, const Pack& pk, int n, int r0, int r1, float* y, int ldy) {
  const int nb = w.cols / kBlk;
  const std::size_t rb = w.row_bytes();
  const auto* base = static_cast<const std::uint8_t*>(w.q);
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
  const int8x16_t s8 = vdupq_n_s8(8);
  thread_local std::vector<std::int8_t> wbuf;
  thread_local std::vector<float> dbuf;
  wbuf.resize(static_cast<std::size_t>(nb) * 128);
  dbuf.resize(static_cast<std::size_t>(nb) * 8);
  auto put = [&](const float32x4_t* o, int r, int p, int tp) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < tp; ++j) {
        const float32x4_t v = o[i * tp + j];
        const int t = 2 * (p + j);
        const int rr = r + 2 * i;
        y[static_cast<std::size_t>(t) * ldy + rr] = vgetq_lane_f32(v, 0);
        y[static_cast<std::size_t>(t) * ldy + rr + 1] = vgetq_lane_f32(v, 2);
        if (t + 1 < n) {
          y[static_cast<std::size_t>(t + 1) * ldy + rr] = vgetq_lane_f32(v, 1);
          y[static_cast<std::size_t>(t + 1) * ldy + rr + 1] = vgetq_lane_f32(v, 3);
        }
      }
    }
  };
  for (int r = r0; r + 4 <= r1; r += 4) {
    for (int b = 0; b < nb; ++b) {
      int8x16_t l[4], h[4];
      for (int k = 0; k < 4; ++k) {
        const std::uint8_t* rp = base + rb * static_cast<std::size_t>(r + k);
        if (w.dt == DType::Q8) {
          const auto* q = reinterpret_cast<const std::int8_t*>(rp) + b * kBlk;
          l[k] = vld1q_s8(q);
          h[k] = vld1q_s8(q + 16);
        } else {
          const uint8x16_t v = vld1q_u8(rp + b * 16);
          l[k] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), s8);
          h[k] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), s8);
        }
      }
      std::int8_t* o = wbuf.data() + static_cast<std::size_t>(b) * 128;
      for (int i = 0; i < 2; ++i) {
        const int64x2_t lo0 = vreinterpretq_s64_s8(l[2 * i]), lo1 = vreinterpretq_s64_s8(l[2 * i + 1]);
        const int64x2_t hi0 = vreinterpretq_s64_s8(h[2 * i]), hi1 = vreinterpretq_s64_s8(h[2 * i + 1]);
        vst1q_s8(o + 64 * i, vreinterpretq_s8_s64(vzip1q_s64(lo0, lo1)));
        vst1q_s8(o + 64 * i + 16, vreinterpretq_s8_s64(vzip2q_s64(lo0, lo1)));
        vst1q_s8(o + 64 * i + 32, vreinterpretq_s8_s64(vzip1q_s64(hi0, hi1)));
        vst1q_s8(o + 64 * i + 48, vreinterpretq_s8_s64(vzip2q_s64(hi0, hi1)));
        const float e0 = h2f(w.s[static_cast<std::size_t>(r + 2 * i) * nb + b]);
        const float e1 = h2f(w.s[static_cast<std::size_t>(r + 2 * i + 1) * nb + b]);
        float* dv = dbuf.data() + b * 8 + 4 * i;
        dv[0] = e0;
        dv[1] = e0;
        dv[2] = e1;
        dv[3] = e1;
      }
    }
    float32x4_t o[8];
    int p = 0;
    while (p + 4 <= pk.pairs) {
      tile<4>(wbuf.data(), dbuf.data(), pk, p, nb, o);
      put(o, r, p, 4);
      p += 4;
    }
    if (p + 2 <= pk.pairs) {
      tile<2>(wbuf.data(), dbuf.data(), pk, p, nb, o);
      put(o, r, p, 2);
      p += 2;
    }
    if (p < pk.pairs) {
      tile<1>(wbuf.data(), dbuf.data(), pk, p, nb, o);
      put(o, r, p, 1);
    }
  }
}

#else

bool has_mm() { return false; }
void pack_act(const Act&, Pack&) {}
void mm_rows(const Tensor&, const Pack&, int, int, int, float*, int) {}

#endif

const char* simd_name() {
#if defined(__ARM_FEATURE_MATMUL_INT8)
  return "neon+dotprod+i8mm";
#elif defined(__ARM_FEATURE_DOTPROD)
  return "neon+dotprod";
#else
  return "neon";
#endif
}

#elif defined(VERBO_AVX2)

namespace {

inline float hsum(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

inline __m256 idot(__m256i w, __m256i x) {
  const __m256i ax = _mm256_sign_epi8(x, w);
  const __m256i aw = _mm256_sign_epi8(w, w);
  const __m256i d16 = _mm256_maddubs_epi16(aw, ax);
  return _mm256_cvtepi32_ps(_mm256_madd_epi16(d16, _mm256_set1_epi16(1)));
}

}

float dot_f32(const float* a, const float* b, int n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0;
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  float r = hsum(_mm256_add_ps(s0, s1));
  for (; i < n; ++i) r += a[i] * b[i];
  return r;
}

float dot_q8(const std::int8_t* w, const std::uint16_t* ws, const std::int8_t* x, const float* xs, int nb) {
  __m256 acc = _mm256_setzero_ps();
  for (int b = 0; b < nb; ++b) {
    const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + b * kBlk));
    const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + b * kBlk));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(ws[b]) * xs[b]), idot(wv, xv), acc);
  }
  return hsum(acc);
}

float dot_q4(const std::uint8_t* w, const std::uint16_t* ws, const std::int8_t* x, const float* xs, int nb) {
  __m256 acc = _mm256_setzero_ps();
  const __m128i m4 = _mm_set1_epi8(0x0F);
  const __m256i s8 = _mm256_set1_epi8(8);
  for (int b = 0; b < nb; ++b) {
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + b * 16));
    const __m128i lo = _mm_and_si128(v, m4);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
    const __m256i wv = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), s8);
    const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + b * kBlk));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(ws[b]) * xs[b]), idot(wv, xv), acc);
  }
  return hsum(acc);
}

const char* simd_name() { return "avx2"; }

#else

float dot_f32(const float* a, const float* b, int n) {
  float r = 0.0f;
  for (int i = 0; i < n; ++i) r += a[i] * b[i];
  return r;
}

float dot_q8(const std::int8_t* w, const std::uint16_t* ws, const std::int8_t* x, const float* xs, int nb) {
  float r = 0.0f;
  for (int b = 0; b < nb; ++b) {
    int s = 0;
    for (int i = 0; i < kBlk; ++i) s += w[b * kBlk + i] * x[b * kBlk + i];
    r += h2f(ws[b]) * xs[b] * static_cast<float>(s);
  }
  return r;
}

float dot_q4(const std::uint8_t* w, const std::uint16_t* ws, const std::int8_t* x, const float* xs, int nb) {
  float r = 0.0f;
  for (int b = 0; b < nb; ++b) {
    int s = 0;
    for (int i = 0; i < 16; ++i) {
      s += ((w[b * 16 + i] & 0x0F) - 8) * x[b * kBlk + i];
      s += ((w[b * 16 + i] >> 4) - 8) * x[b * kBlk + i + 16];
    }
    r += h2f(ws[b]) * xs[b] * static_cast<float>(s);
  }
  return r;
}

const char* simd_name() { return "scalar"; }

#endif


#if !defined(VERBO_NEON)

struct Pack {
  int pairs = 0;
};

void rows1(const Tensor& w, const std::int8_t* x, const float* xs, int r0, int r1, float* y) {
  const int nb = w.cols / kBlk;
  const std::size_t rb = w.row_bytes();
  const auto* base = static_cast<const std::uint8_t*>(w.q);
  for (int r = r0; r < r1; ++r) {
    const std::uint8_t* p = base + rb * r;
    const std::uint16_t* sc = w.s + static_cast<std::size_t>(r) * nb;
    y[r] = w.dt == DType::Q8 ? dot_q8(reinterpret_cast<const std::int8_t*>(p), sc, x, xs, nb) : dot_q4(p, sc, x, xs, nb);
  }
}

bool has_mm() { return false; }
void pack_act(const Act&, Pack&) {}
void mm_rows(const Tensor&, const Pack&, int, int, int, float*, int) {}

#endif

void matmul(const Tensor& w, const Act& x, float* y, int ldy, Pool& pool) {
  const int rows = w.rows;
  const int cols = w.cols;
  const int nth = pool.size();
  int chunk;
  if (x.n == 1) {
    chunk = std::max(4, static_cast<int>(65536 / std::max<std::size_t>(1, w.row_bytes())) & ~3);
    chunk = std::min(chunk, std::max(4, ((rows + nth - 1) / nth + 3) & ~3));
  } else {
    chunk = std::max(4, (rows + nth * 8 - 1) / (nth * 8));
    chunk = (chunk + 3) & ~3;
  }
  const int tasks = (rows + chunk - 1) / chunk;
  const std::size_t rb = w.row_bytes();
  const bool quant = w.dt != DType::F32;
  const bool mm = quant && x.n >= 2 && has_mm();
  thread_local Pack tl_pk;
  Pack& pk = tl_pk;
  if (mm) pack_act(x, pk);
  pool.run(tasks, [&](int t, int) {
    const int r0 = t * chunk;
    const int r1 = std::min(rows, r0 + chunk);
    if (quant) {
      int rs = r0;
      if (mm) {
        rs = r0 + (r1 - r0) / 4 * 4;
        mm_rows(w, pk, x.n, r0, rs, y, ldy);
      }
      if (rs < r1) {
        for (int i = 0; i < x.n; ++i) {
          rows1(w, x.q + static_cast<std::size_t>(i) * x.ld, x.s + static_cast<std::size_t>(i) * (x.ld / kBlk), rs, r1,
                y + static_cast<std::size_t>(i) * ldy);
        }
      }
      return;
    }
    for (int r = r0; r < r1; ++r) {
      const auto* wr = reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(w.q) + rb * static_cast<std::size_t>(r));
      for (int i = 0; i < x.n; ++i) {
        y[static_cast<std::size_t>(i) * ldy + r] = dot_f32(wr, x.f + static_cast<std::size_t>(i) * x.ld, cols);
      }
    }
  });
}

}
