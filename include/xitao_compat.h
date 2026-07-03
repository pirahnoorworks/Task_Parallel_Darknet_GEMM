#ifndef XITAO_COMPAT_H
#define XITAO_COMPAT_H

#if defined(USE_NATIVE_XITAO)
#include <xitao.h>
#else

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#ifndef XITAO_MAXTHREADS
#define XITAO_MAXTHREADS 64
#endif

class AssemblyTask {
 public:
  explicit AssemblyTask(int taskWidth)
      : width(taskWidth), criticality(0), leader(0), remaining_predecessors(0) {}

  virtual ~AssemblyTask() = default;
  virtual int execute(int threadid) = 0;
  virtual int cleanup() { return 0; }

  void make_edge(AssemblyTask* next) {
    next->remaining_predecessors.fetch_add(1, std::memory_order_relaxed);
    successors.push_back(next);
  }

  template <typename TableType>
  void print_ptt(TableType, const char*) {}

  int width;
  int criticality;
  int leader;

 private:
  std::vector<AssemblyTask*> successors;
  std::atomic<int> remaining_predecessors;

  friend class XiTAORuntime;
};

class XiTAORuntime {
 public:
  static XiTAORuntime& instance() {
    static XiTAORuntime runtime;
    return runtime;
  }

  void init(int threads = 0) {
    std::lock_guard<std::mutex> lock(mu_);
    if (threads > 0) {
      configured_threads_ = threads;
    } else if (configured_threads_ == 0) {
      configured_threads_ = static_cast<int>(std::thread::hardware_concurrency());
      if (configured_threads_ <= 0) {
        configured_threads_ = 1;
      }
    }
  }

  void push(AssemblyTask* task, int) {
    if (task->remaining_predecessors.load(std::memory_order_relaxed) != 0) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      ready_.push(task);
    }
    cv_.notify_one();
  }

  void start() {
    std::lock_guard<std::mutex> lock(mu_);
    if (started_) {
      return;
    }
    shutdown_ = false;
    const int threadCount = configured_threads_ > 0 ? configured_threads_ : 1;
    workers_.reserve(static_cast<size_t>(threadCount));
    for (int i = 0; i < threadCount; ++i) {
      workers_.emplace_back([this, i]() { workerLoop(i); });
    }
    started_ = true;
  }

  void fini() {
    {
      std::unique_lock<std::mutex> lock(mu_);
      idle_cv_.wait(lock, [this]() { return ready_.empty() && active_tasks_ == 0; });
      shutdown_ = true;
    }
    cv_.notify_all();

    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      workers_.clear();
      started_ = false;
      shutdown_ = false;
    }
  }

 private:
  XiTAORuntime() : configured_threads_(0), started_(false), shutdown_(false), active_tasks_(0) {}

  void workerLoop(int workerId) {
    while (true) {
      AssemblyTask* task = nullptr;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this]() { return shutdown_ || !ready_.empty(); });

        if (shutdown_ && ready_.empty()) {
          return;
        }

        task = ready_.front();
        ready_.pop();
        ++active_tasks_;
      }

      task->execute(workerId);
      task->cleanup();

      {
        std::lock_guard<std::mutex> lock(mu_);
        for (AssemblyTask* successor : task->successors) {
          const int remaining =
              successor->remaining_predecessors.fetch_sub(1, std::memory_order_acq_rel) - 1;
          if (remaining == 0) {
            ready_.push(successor);
          }
        }
        --active_tasks_;
        if (ready_.empty() && active_tasks_ == 0) {
          idle_cv_.notify_all();
        }
      }
      cv_.notify_all();
    }
  }

  int configured_threads_;
  bool started_;
  bool shutdown_;
  int active_tasks_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable idle_cv_;
  std::queue<AssemblyTask*> ready_;
  std::vector<std::thread> workers_;
};

inline void gotao_init(int threads = 0) {
  XiTAORuntime::instance().init(threads);
}

inline void gotao_push(AssemblyTask* task, int queueId) {
  XiTAORuntime::instance().push(task, queueId);
}

inline void gotao_start() {
  XiTAORuntime::instance().start();
}

inline void gotao_fini() {
  XiTAORuntime::instance().fini();
}

#endif

#endif
