#include "darknet_gemm.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "xitao_compat.h"

namespace {

struct TaskInput {
  const std::vector<float>* weightsA;
  const std::vector<float>* im2colB;
  std::vector<float>* outputC;
  int n;
  int k;
  int rowStart;
  int rowEnd;
};

class GT1Task : public AssemblyTask {
 public:
  explicit GT1Task(TaskInput input) : AssemblyTask(1), input_(input) {}

  int execute(int) override {
    std::vector<float>& c = *input_.outputC;
    for (int row = input_.rowStart; row < input_.rowEnd; ++row) {
      const int base = row * input_.n;
      for (int col = 0; col < input_.n; ++col) {
        c[base + col] = 0.0f;
      }
    }
    return 0;
  }

 private:
  TaskInput input_;
};

class GT2Task : public AssemblyTask {
 public:
  explicit GT2Task(TaskInput input) : AssemblyTask(1), input_(input) {}

  int execute(int) override {
    const std::vector<float>& a = *input_.weightsA;
    const std::vector<float>& b = *input_.im2colB;
    std::vector<float>& c = *input_.outputC;

    for (int row = input_.rowStart; row < input_.rowEnd; ++row) {
      for (int kk = 0; kk < input_.k; ++kk) {
        const float apart = a[row * input_.k + kk];
        const int bBase = kk * input_.n;
        const int cBase = row * input_.n;
        for (int col = 0; col < input_.n; ++col) {
          c[cBase + col] += apart * b[bBase + col];
        }
      }
    }
    return 0;
  }

 private:
  TaskInput input_;
};

int clamp_block_rows(int m, int blockRows) {
  if (blockRows <= 0) {
    return std::max(1, std::min(64, m));
  }
  return std::max(1, std::min(blockRows, m));
}

float im2col_get_pixel(const std::vector<float>& image,
                       int height,
                       int width,
                       int channels,
                       int row,
                       int col,
                       int channel,
                       int pad) {
  row -= pad;
  col -= pad;

  if (row < 0 || col < 0 || row >= height || col >= width) {
    return 0.0f;
  }
  return image[static_cast<size_t>(col + width * (row + height * channel))];
}

}  // namespace

DarknetGemmProblem problem_from_conv(const ConvLayerConfig& cfg) {
  const int outH = (cfg.height + 2 * cfg.pad - cfg.kernel) / cfg.stride + 1;
  const int outW = (cfg.width + 2 * cfg.pad - cfg.kernel) / cfg.stride + 1;

  return DarknetGemmProblem{
      cfg.filters,
      outH * outW,
      cfg.channels * cfg.kernel * cfg.kernel,
      64,
      8};
}

void im2col_cpu(const std::vector<float>& image,
                const ConvLayerConfig& cfg,
                std::vector<float>& dataCol) {
  const int heightCol = (cfg.height + 2 * cfg.pad - cfg.kernel) / cfg.stride + 1;
  const int widthCol = (cfg.width + 2 * cfg.pad - cfg.kernel) / cfg.stride + 1;
  const int channelsCol = cfg.channels * cfg.kernel * cfg.kernel;

  assert(static_cast<int>(image.size()) == cfg.channels * cfg.height * cfg.width);
  assert(static_cast<int>(dataCol.size()) == channelsCol * heightCol * widthCol);

  for (int c = 0; c < channelsCol; ++c) {
    const int wOffset = c % cfg.kernel;
    const int hOffset = (c / cfg.kernel) % cfg.kernel;
    const int cIm = c / (cfg.kernel * cfg.kernel);

    for (int h = 0; h < heightCol; ++h) {
      for (int w = 0; w < widthCol; ++w) {
        const int imRow = hOffset + h * cfg.stride;
        const int imCol = wOffset + w * cfg.stride;
        const int colIndex = (c * heightCol + h) * widthCol + w;

        dataCol[static_cast<size_t>(colIndex)] =
            im2col_get_pixel(image, cfg.height, cfg.width, cfg.channels, imRow, imCol, cIm, cfg.pad);
      }
    }
  }
}

void fill_deterministic(std::vector<float>& data, float scale, float bias) {
  for (size_t i = 0; i < data.size(); ++i) {
    const float value = static_cast<float>((i % 97) - 48) / 48.0f;
    data[i] = scale * value + bias;
  }
}

void gemm_nt_openmp(const DarknetGemmProblem& problem,
                    const std::vector<float>& weightsA,
                    const std::vector<float>& im2colB,
                    std::vector<float>& outputC) {
  assert(static_cast<int>(weightsA.size()) == problem.m * problem.k);
  assert(static_cast<int>(im2colB.size()) == problem.k * problem.n);
  assert(static_cast<int>(outputC.size()) == problem.m * problem.n);

  #pragma omp parallel for schedule(static)
  for (int row = 0; row < problem.m; ++row) {
    const int cBase = row * problem.n;
    for (int col = 0; col < problem.n; ++col) {
      outputC[static_cast<size_t>(cBase + col)] = 0.0f;
    }
  }

  #pragma omp parallel for schedule(static)
  for (int row = 0; row < problem.m; ++row) {
    for (int kk = 0; kk < problem.k; ++kk) {
      const float apart = weightsA[row * problem.k + kk];
      const int bBase = kk * problem.n;
      const int cBase = row * problem.n;
      for (int col = 0; col < problem.n; ++col) {
        outputC[cBase + col] += apart * im2colB[bBase + col];
      }
    }
  }
}

void gemm_nt_xitao(const DarknetGemmProblem& problem,
                   const std::vector<float>& weightsA,
                   const std::vector<float>& im2colB,
                   std::vector<float>& outputC) {
  assert(static_cast<int>(weightsA.size()) == problem.m * problem.k);
  assert(static_cast<int>(im2colB.size()) == problem.k * problem.n);
  assert(static_cast<int>(outputC.size()) == problem.m * problem.n);

  const int blockRows = clamp_block_rows(problem.m, problem.taskBlockRows);
  const int numTasks = (problem.m + blockRows - 1) / blockRows;

  std::vector<std::unique_ptr<GT1Task>> gt1Tasks;
  std::vector<std::unique_ptr<GT2Task>> gt2Tasks;
  gt1Tasks.reserve(static_cast<size_t>(numTasks));
  gt2Tasks.reserve(static_cast<size_t>(numTasks));

  gotao_init(problem.runtimeThreads);

  for (int taskIdx = 0; taskIdx < numTasks; ++taskIdx) {
    const int rowStart = taskIdx * blockRows;
    const int rowEnd = std::min(problem.m, rowStart + blockRows);

    TaskInput input{&weightsA, &im2colB, &outputC, problem.n, problem.k, rowStart, rowEnd};

    gt1Tasks.emplace_back(std::make_unique<GT1Task>(input));
    gt2Tasks.emplace_back(std::make_unique<GT2Task>(input));

    gt1Tasks.back()->make_edge(gt2Tasks.back().get());

    const int queue = problem.runtimeThreads > 0 ? taskIdx % problem.runtimeThreads : 0;
    gotao_push(gt1Tasks.back().get(), queue);
  }

  gotao_start();
  gotao_fini();
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  assert(lhs.size() == rhs.size());
  float maxDiff = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    maxDiff = std::max(maxDiff, std::fabs(lhs[i] - rhs[i]));
  }
  return maxDiff;
}
