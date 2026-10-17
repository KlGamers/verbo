#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace verbo {

class Pool {
 public:
  using Fn = std::function<void(int task, int tid)>;

  explicit Pool(int n);
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  int size() const { return static_cast<int>(wk_.size()) + 1; }
  void run(int tasks, const Fn& fn);

  static int default_threads();

 private:
  std::vector<std::thread> wk_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<std::uint64_t> gen_{0};
  std::atomic<int> next_{0};
  std::atomic<int> busy_{0};
  std::atomic<bool> stop_{false};
  std::atomic<int> sleepers_{0};
  int tasks_ = 0;
  const Fn* fn_ = nullptr;

  void loop(int tid);
  void work(int tid);
};

}
