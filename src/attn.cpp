#include "attn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(VERBO_FORCE_SCALAR)
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define VERBO_NEON 1
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define VERBO_AVX2 1
#endif

namespace verbo {

namespace {

constexpr int KB = kKvBlk;
#if defined(VERBO_NEON) || defined(VERBO_AVX2)
constexpr float kC1 = 0.693359375f;
constexpr float kC2 = -2.12194440e-4f;
constexpr float kLog2e = 1.44269504088896341f;
#endif

#if defined(VERBO_NEON)

inline float32x4_t vexp(float32x4_t x) {
  const uint32x4_t tiny = vcltq_f32(x, vdupq_n_f32(-87.3f));
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.3f));
  const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, kLog2e));
  float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(kC1));
  r = vfmsq_f32(r, n, vdupq_n_f32(kC2));
  float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
  p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
  p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
  p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
  p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
  const float32x4_t y = vaddq_f32(vfmaq_f32(r, p, vmulq_f32(r, r)), vdupq_n_f32(1.0f));
  const int32x4_t e = vshlq_n_s32(vcvtq_s32_f32(n), 23);
  return vbslq_f32(tiny, vdupq_n_f32(0.0f), vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(y), e)));
}

#elif defined(VERBO_AVX2)

inline __m256 vexp(__m256 x) {
  const __m256 tiny = _mm256_cmp_ps(x, _mm256_set1_ps(-87.3f), _CMP_LT_OQ);
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
  const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(kLog2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(kC1), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(kC2), r);
  __m256 p = _mm256_set1_ps(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
  const __m256 y = _mm256_add_ps(_mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r), _mm256_set1_ps(1.0f));
  const __m256i e = _mm256_slli_epi32(_mm256_cvtps_epi32(n), 23);
  return _mm256_andnot_ps(tiny, _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(y), e)));
}

#endif

void scores(const float* q, const float* kt, int hd, float scale, float* s) {
#if defined(VERBO_NEON)
  float32x4_t acc[16];
  for (int k = 0; k < 16; ++k) acc[k] = vdupq_n_f32(0.0f);
  for (int d = 0; d < hd; ++d) {
    const float32x4_t qd = vdupq_n_f32(q[d] * scale);
    const float* kd = kt + d * KB;
#pragma clang loop unroll(full)
    for (int k = 0; k < 16; ++k) acc[k] = vfmaq_f32(acc[k], vld1q_f32(kd + 4 * k), qd);
  }
  for (int k = 0; k < 16; ++k) vst1q_f32(s + 4 * k, acc[k]);
#else
  std::fill(s, s + KB, 0.0f);
  for (int d = 0; d < hd; ++d) {
    const float qd = q[d] * scale;
    const float* kd = kt + d * KB;
    for (int j = 0; j < KB; ++j) s[j] += qd * kd[j];
  }
#endif
}

void pv(const float* p, int lim, const float* v, int kvld, int hd, float corr, float* o) {
#if defined(VERBO_NEON)
  int dc = 0;
  for (; dc + 64 <= hd; dc += 64) {
    float32x4_t acc[16];
    const float32x4_t cv = vdupq_n_f32(corr);
#pragma clang loop unroll(full)
    for (int k = 0; k < 16; ++k) acc[k] = vmulq_f32(vld1q_f32(o + dc + 4 * k), cv);
    for (int j = 0; j < lim; ++j) {
      const float32x4_t pj = vdupq_n_f32(p[j]);
      const float* vr = v + static_cast<std::size_t>(j) * kvld + dc;
#pragma clang loop unroll(full)
      for (int k = 0; k < 16; ++k) acc[k] = vfmaq_f32(acc[k], vld1q_f32(vr + 4 * k), pj);
    }
    for (int k = 0; k < 16; ++k) vst1q_f32(o + dc + 4 * k, acc[k]);
  }
  if (dc == hd) return;
  for (int d = dc; d < hd; ++d) o[d] *= corr;
  for (int j = 0; j < lim; ++j) {
    const float* vr = v + static_cast<std::size_t>(j) * kvld;
    for (int d = dc; d < hd; ++d) o[d] += p[j] * vr[d];
  }
#else
  for (int d = 0; d < hd; ++d) o[d] *= corr;
  for (int j = 0; j < lim; ++j) {
    const float* vr = v + static_cast<std::size_t>(j) * kvld;
    for (int d = 0; d < hd; ++d) o[d] += p[j] * vr[d];
  }
#endif
}

}

void exp_inplace(float* x, int n) {
  int i = 0;
#if defined(VERBO_NEON)
  for (; i + 4 <= n; i += 4) vst1q_f32(x + i, vexp(vld1q_f32(x + i)));
#elif defined(VERBO_AVX2)
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, vexp(_mm256_loadu_ps(x + i)));
#endif
  for (; i < n; ++i) x[i] = std::exp(x[i]);
}

void silu_mul(const float* g, const float* u, float* out, int n) {
  int i = 0;
#if defined(VERBO_NEON)
  const float32x4_t one = vdupq_n_f32(1.0f);
  for (; i + 4 <= n; i += 4) {
    const float32x4_t gv = vld1q_f32(g + i);
    const float32x4_t e = vexp(vnegq_f32(gv));
    vst1q_f32(out + i, vdivq_f32(vmulq_f32(gv, vld1q_f32(u + i)), vaddq_f32(one, e)));
  }
#elif defined(VERBO_AVX2)
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 zero = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    const __m256 gv = _mm256_loadu_ps(g + i);
    const __m256 e = vexp(_mm256_sub_ps(zero, gv));
    _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_mul_ps(gv, _mm256_loadu_ps(u + i)), _mm256_add_ps(one, e)));
  }
#endif
  for (; i < n; ++i) out[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
}

void attention(const AttnArgs& a, Pool& pool, AttnScratch& sc) {
  const int grp = a.heads / a.kv_heads;
  const int hd = a.hd;
  const int total = a.p0 + a.n;
  const int qb = std::min(a.n, 8);
  const int nqb = (a.n + qb - 1) / qb;
  const int rows = qb * grp;
  const int nth = pool.size();
  const int base = a.kv_heads * nqb;
  int splits = 1;
  if (base < 2 * nth && total > 2 * KB) {
    splits = std::min((2 * nth + base - 1) / base, (total + KB - 1) / KB);
  }
  const int per = ((total + splits - 1) / splits + KB - 1) / KB * KB;
  const std::size_t rec = static_cast<std::size_t>(hd) + 2;
  if (splits > 1) sc.part.resize(static_cast<std::size_t>(splits) * base * rows * rec);
  const std::size_t need = KB + static_cast<std::size_t>(rows) * (hd + 2);
  if (sc.th.size() < static_cast<std::size_t>(nth)) sc.th.resize(static_cast<std::size_t>(nth));
  for (auto& t : sc.th) {
    if (t.size() < need) t.resize(need);
  }
  pool.run(base * splits, [&](int t, int tid) {
    const int g = t % a.kv_heads;
    const int rest = t / a.kv_heads;
    const int qi = rest % nqb;
    const int sp = rest / nqb;
    const int i0 = qi * qb, i1 = std::min(a.n, i0 + qb);
    const int nr = (i1 - i0) * grp;
    const int ks = sp * per;
    const int ke = std::min({total, ks + per, a.p0 + i1});
    float* s = sc.th[static_cast<std::size_t>(tid)].data();
    float* o = s + KB;
    float* m = o + static_cast<std::size_t>(rows) * hd;
    float* l = m + rows;
    std::fill(o, o + static_cast<std::size_t>(nr) * hd, 0.0f);
    std::fill(m, m + nr, -INFINITY);
    std::fill(l, l + nr, 0.0f);
    for (int kb = ks; kb < ke; kb += KB) {
      const int kc = std::min(KB, ke - kb);
      const float* kt = a.kt + g * a.kth + static_cast<std::size_t>(kb / KB) * hd * KB;
      for (int r = 0; r < nr; ++r) {
        const int i = i0 + r / grp;
        const int h = g * grp + r % grp;
        const int lim = std::min(kb + kc, a.p0 + i + 1) - kb;
        if (lim <= 0) continue;
        const float* q = a.q + static_cast<std::size_t>(i) * a.qld + h * hd;
        scores(q, kt, hd, a.scale, s);
        float mx = m[r];
        for (int j = 0; j < lim; ++j) mx = std::max(mx, s[j]);
        for (int j = 0; j < lim; ++j) s[j] -= mx;
        exp_inplace(s, lim);
        float sum = 0.0f;
        for (int j = 0; j < lim; ++j) sum += s[j];
        const float corr = m[r] == -INFINITY ? 0.0f : std::exp(m[r] - mx);
        l[r] = l[r] * corr + sum;
        m[r] = mx;
        pv(s, lim, a.v + static_cast<std::size_t>(kb) * a.kvld + g * hd, a.kvld, hd, corr,
           o + static_cast<std::size_t>(r) * hd);
      }
    }
    for (int r = 0; r < nr; ++r) {
      const int i = i0 + r / grp;
      const int h = g * grp + r % grp;
      const float* orow = o + static_cast<std::size_t>(r) * hd;
      if (splits == 1) {
        float* dst = a.out + static_cast<std::size_t>(i) * a.old + h * hd;
        const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        for (int d = 0; d < hd; ++d) dst[d] = orow[d] * inv;
      } else {
        float* p = sc.part.data() + ((static_cast<std::size_t>(sp) * base + qi * a.kv_heads + g) * rows + r) * rec;
        p[0] = m[r];
        p[1] = l[r];
        std::memcpy(p + 2, orow, sizeof(float) * hd);
      }
    }
  });
  if (splits == 1) return;
  for (int qi = 0; qi < nqb; ++qi) {
    const int i0 = qi * qb, i1 = std::min(a.n, i0 + qb);
    const int nr = (i1 - i0) * grp;
    for (int g = 0; g < a.kv_heads; ++g) {
      for (int r = 0; r < nr; ++r) {
        const int i = i0 + r / grp;
        const int h = g * grp + r % grp;
        float mx = -INFINITY;
        for (int sp = 0; sp < splits; ++sp) {
          const float* p = sc.part.data() + ((static_cast<std::size_t>(sp) * base + qi * a.kv_heads + g) * rows + r) * rec;
          if (p[1] > 0.0f) mx = std::max(mx, p[0]);
        }
        float* dst = a.out + static_cast<std::size_t>(i) * a.old + h * hd;
        std::fill(dst, dst + hd, 0.0f);
        float lt = 0.0f;
        for (int sp = 0; sp < splits; ++sp) {
          const float* p = sc.part.data() + ((static_cast<std::size_t>(sp) * base + qi * a.kv_heads + g) * rows + r) * rec;
          if (p[1] <= 0.0f) continue;
          const float w = std::exp(p[0] - mx);
          lt += p[1] * w;
          for (int d = 0; d < hd; ++d) dst[d] += p[2 + d] * w;
        }
        const float inv = lt > 0.0f ? 1.0f / lt : 0.0f;
        for (int d = 0; d < hd; ++d) dst[d] *= inv;
      }
    }
  }
}

}
