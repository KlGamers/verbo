#include "pool.hpp"

#include <algorithm>
#include <chrono>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define VERBO_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define VERBO_PAUSE() __asm__ __volatile__("yield")
#else
#define VERBO_PAUSE() ((void)0)
#endif

namespace verbo {

namespace {

void fast_qos() {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

}

Pool::Pool(int n) {
  n = std::max(1, n);
  fast_qos();
  for (int i = 1; i < n; ++i) wk_.emplace_back([this, i] { loop(i); });
}

Pool::~Pool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_.store(true);
    gen_.fetch_add(1);
  }
  cv_.notify_all();
  for (auto& t : wk_) t.join();
}

int Pool::default_threads() {
#if defined(__APPLE__)
  int perf = 0;
  std::size_t len = sizeof perf;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, nullptr, 0) == 0 && perf > 0) return perf;
#endif
  const unsigned hw = std::thread::hardware_concurrency();
  return hw ? static_cast<int>(std::max(1u, hw / 2)) : 4;
}

void Pool::work(int tid) {
  for (;;) {
    const int t = next_.fetch_add(1, std::memory_order_relaxed);
    if (t >= tasks_) break;
    (*fn_)(t, tid);
  }
}

void Pool::loop(int tid) {
  fast_qos();
  std::uint64_t seen = 0;
  for (;;) {
    std::uint64_t g = gen_.load(std::memory_order_acquire);
    if (g == seen) {
      const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
      int spin = 0;
      while ((g = gen_.load(std::memory_order_acquire)) == seen) {
        VERBO_PAUSE();
        if (++spin == 256) {
          spin = 0;
          if (std::chrono::steady_clock::now() > until) break;
        }
      }
      if (g == seen) {
        std::unique_lock<std::mutex> lk(mu_);
        sleepers_.fetch_add(1);
        cv_.wait(lk, [&] { return gen_.load() != seen; });
        sleepers_.fetch_sub(1);
        g = gen_.load(std::memory_order_acquire);
      }
    }
    seen = g;
    if (stop_.load()) return;
    work(tid);
    busy_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

void Pool::run(int tasks, const Fn& fn) {
  if (tasks <= 0) return;
  if (wk_.empty() || tasks == 1) {
    for (int t = 0; t < tasks; ++t) fn(t, 0);
    return;
  }
  fn_ = &fn;
  tasks_ = tasks;
  next_.store(0, std::memory_order_relaxed);
  busy_.store(static_cast<int>(wk_.size()), std::memory_order_relaxed);
  if (sleepers_.load() > 0) {
    std::lock_guard<std::mutex> lk(mu_);
    gen_.fetch_add(1);
    cv_.notify_all();
  } else {
    gen_.fetch_add(1);
    if (sleepers_.load() > 0) {
      std::lock_guard<std::mutex> lk(mu_);
      cv_.notify_all();
    }
  }
  work(0);
  while (busy_.load(std::memory_order_acquire) > 0) VERBO_PAUSE();
}

}
