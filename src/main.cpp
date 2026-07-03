#include <chrono>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "darknet_gemm.h"

namespace {

enum class Backend {
  OpenMP,
  Xitao,
  Both
};

struct Config {
  int channels = 64;
  int height = 56;
  int width = 56;
  int kernel = 3;
  int stride = 1;
  int pad = 1;
  int filters = 64;
  int blockRows = 64;
  int threads = 8;
  int warmup = 2;
  int iters = 10;
  bool verify = true;
  Backend backend = Backend::Both;
};

bool readInt(const std::vector<std::string>& args,
             size_t& i,
             const std::string& flag,
             int& out) {
  if (args[i] != flag) {
    return false;
  }
  if (i + 1 >= args.size()) {
    throw std::runtime_error("Missing value for " + flag);
  }
  out = std::stoi(args[++i]);
  return true;
}

Backend parseBackend(const std::string& value) {
  if (value == "omp") {
    return Backend::OpenMP;
  }
  if (value == "xitao") {
    return Backend::Xitao;
  }
  if (value == "both") {
    return Backend::Both;
  }
  throw std::runtime_error("Invalid backend: " + value + ". Use omp|xitao|both.");
}

Config parseArgs(int argc, char** argv) {
  Config cfg;
  std::vector<std::string> args(argv + 1, argv + argc);

  for (size_t i = 0; i < args.size(); ++i) {
    if (readInt(args, i, "--channels", cfg.channels)) {
      continue;
    }
    if (readInt(args, i, "--height", cfg.height)) {
      continue;
    }
    if (readInt(args, i, "--width", cfg.width)) {
      continue;
    }
    if (readInt(args, i, "--kernel", cfg.kernel)) {
      continue;
    }
    if (readInt(args, i, "--stride", cfg.stride)) {
      continue;
    }
    if (readInt(args, i, "--pad", cfg.pad)) {
      continue;
    }
    if (readInt(args, i, "--filters", cfg.filters)) {
      continue;
    }
    if (readInt(args, i, "--block-rows", cfg.blockRows)) {
      continue;
    }
    if (readInt(args, i, "--threads", cfg.threads)) {
      continue;
    }
    if (readInt(args, i, "--warmup", cfg.warmup)) {
      continue;
    }
    if (readInt(args, i, "--iters", cfg.iters)) {
      continue;
    }
    if (args[i] == "--no-verify") {
      cfg.verify = false;
      continue;
    }
    if (args[i] == "--backend") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("Missing value for --backend.");
      }
      cfg.backend = parseBackend(args[++i]);
      continue;
    }
    if (args[i] == "--help" || args[i] == "-h") {
      std::cout
          << "Usage: darknet_xitao_dgemm_demo [options]\n"
          << "  --channels <int>    Input channels C, default 64\n"
          << "  --height <int>      Input height H, default 56\n"
          << "  --width <int>       Input width W, default 56\n"
          << "  --kernel <int>      Kernel size R=S, default 3\n"
          << "  --stride <int>      Conv stride, default 1\n"
          << "  --pad <int>         Conv padding, default 1\n"
          << "  --filters <int>     Number of output filters K, default 64\n"
          << "  --block-rows <int>  Rows per TAO block, default 64\n"
          << "  --threads <int>     XiTAO worker threads, default 8\n"
          << "  --warmup <int>      Warmup iterations, default 2\n"
          << "  --iters <int>       Timed iterations, default 10\n"
          << "  --backend <name>    omp|xitao|both (default both)\n"
          << "  --no-verify         Skip omp-vs-xitao correctness check\n";
      std::exit(0);
    }
    throw std::runtime_error("Unknown argument: " + args[i]);
  }

  return cfg;
}

void validateConfig(const Config& cfg) {
  if (cfg.channels <= 0 || cfg.height <= 0 || cfg.width <= 0 || cfg.kernel <= 0 ||
      cfg.stride <= 0 || cfg.pad < 0 || cfg.filters <= 0 || cfg.blockRows <= 0 ||
      cfg.threads <= 0 || cfg.warmup < 0 || cfg.iters <= 0) {
    throw std::runtime_error("All dimensions/iteration counts must be positive (warmup >= 0).");
  }

  const int outH = (cfg.height + 2 * cfg.pad - cfg.kernel) / cfg.stride + 1;
  const int outW = (cfg.width + 2 * cfg.pad - cfg.kernel) / cfg.stride + 1;
  if (outH <= 0 || outW <= 0) {
    throw std::runtime_error("Invalid convolution geometry: output spatial size must be positive.");
  }
}

using KernelFn = std::function<void(const DarknetGemmProblem&,
                                    const std::vector<float>&,
                                    const std::vector<float>&,
                                    std::vector<float>&)>;

double benchmarkMs(const Config& cfg,
                   const DarknetGemmProblem& problem,
                   const std::vector<float>& a,
                   const std::vector<float>& b,
                   std::vector<float>& c,
                   const KernelFn& kernel) {
  for (int i = 0; i < cfg.warmup; ++i) {
    kernel(problem, a, b, c);
  }

  const auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < cfg.iters; ++i) {
    kernel(problem, a, b, c);
  }
  const auto end = std::chrono::high_resolution_clock::now();

  const std::chrono::duration<double, std::milli> total = end - start;
  return total.count() / static_cast<double>(cfg.iters);
}

void printSummaryHeader(const Config& cfg, const DarknetGemmProblem& problem) {
  const int outH = (cfg.height + 2 * cfg.pad - cfg.kernel) / cfg.stride + 1;
  const int outW = (cfg.width + 2 * cfg.pad - cfg.kernel) / cfg.stride + 1;

  std::cout << "Darknet CNN Layer DGEMM Demo (OpenMP vs XiTAO)\n";
  std::cout << "Conv config N=1, C=" << cfg.channels << ", H=" << cfg.height << ", W=" << cfg.width
            << ", K=" << cfg.filters << ", R=S=" << cfg.kernel << ", stride=" << cfg.stride
            << ", pad=" << cfg.pad << "\n";
  std::cout << "im2col output (outH,outW): (" << outH << ", " << outW << ")\n";
  std::cout << "DGEMM view (M,N,K): (" << problem.m << ", " << problem.n << ", " << problem.k << ")\n";
  std::cout << "TAO block rows: " << problem.taskBlockRows
            << ", XiTAO threads: " << problem.runtimeThreads << "\n";
  std::cout << "Warmup: " << cfg.warmup << ", Timed iterations: " << cfg.iters << "\n";
}

void printPerformanceLine(const std::string& label,
                          double avgMs,
                          const DarknetGemmProblem& problem) {
  const double ops = 2.0 * static_cast<double>(problem.m) * static_cast<double>(problem.n) *
                     static_cast<double>(problem.k);
  const double gflops = (ops / 1e9) / (avgMs / 1000.0);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << label << " avg latency: " << avgMs << " ms"
            << ", throughput: " << gflops << " GFLOP/s\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parseArgs(argc, argv);
    validateConfig(cfg);

    ConvLayerConfig convCfg{cfg.channels, cfg.height, cfg.width, cfg.kernel, cfg.stride, cfg.pad, cfg.filters};
    DarknetGemmProblem problem = problem_from_conv(convCfg);
    problem.taskBlockRows = cfg.blockRows;
    problem.runtimeThreads = cfg.threads;

    std::vector<float> image(static_cast<size_t>(cfg.channels) * static_cast<size_t>(cfg.height) * static_cast<size_t>(cfg.width));
    std::vector<float> weightsA(static_cast<size_t>(problem.m) * static_cast<size_t>(problem.k));
    std::vector<float> im2colB(static_cast<size_t>(problem.k) * static_cast<size_t>(problem.n));
    std::vector<float> outputOmp(static_cast<size_t>(problem.m) * static_cast<size_t>(problem.n), 0.0f);
    std::vector<float> outputXitao(static_cast<size_t>(problem.m) * static_cast<size_t>(problem.n), 0.0f);

    fill_deterministic(image, 0.75f, 0.0f);
    fill_deterministic(weightsA, 0.25f, 0.1f);
    im2col_cpu(image, convCfg, im2colB);

    printSummaryHeader(cfg, problem);

    if (cfg.verify) {
      gemm_nt_openmp(problem, weightsA, im2colB, outputOmp);
      gemm_nt_xitao(problem, weightsA, im2colB, outputXitao);
      const float diff = max_abs_diff(outputOmp, outputXitao);
      const bool pass = diff < 1e-4f;
      std::cout << "Verification max |omp-xitao|: " << diff << " -> "
                << (pass ? "PASSED" : "FAILED") << "\n";
      if (!pass) {
        return 2;
      }
    }

    if (cfg.backend == Backend::OpenMP || cfg.backend == Backend::Both) {
      const double ompMs =
          benchmarkMs(cfg, problem, weightsA, im2colB, outputOmp, gemm_nt_openmp);
      printPerformanceLine("OpenMP gemm_nt", ompMs, problem);
    }

    if (cfg.backend == Backend::Xitao || cfg.backend == Backend::Both) {
      const double xitaoMs =
          benchmarkMs(cfg, problem, weightsA, im2colB, outputXitao, gemm_nt_xitao);
      printPerformanceLine("XiTAO-task gemm_nt", xitaoMs, problem);
    }

    std::cout << "Demo complete.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
