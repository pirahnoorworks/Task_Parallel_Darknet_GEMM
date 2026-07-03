#ifndef DARKNET_GEMM_H
#define DARKNET_GEMM_H

#include <string>
#include <vector>

struct DarknetGemmProblem {
  int m;
  int n;
  int k;
  int taskBlockRows;
  int runtimeThreads;
};

struct ConvLayerConfig {
  int channels;
  int height;
  int width;
  int kernel;
  int stride;
  int pad;
  int filters;
};

DarknetGemmProblem problem_from_conv(const ConvLayerConfig& cfg);

void im2col_cpu(const std::vector<float>& image,
                const ConvLayerConfig& cfg,
                std::vector<float>& dataCol);

void fill_deterministic(std::vector<float>& data, float scale, float bias);

void gemm_nt_openmp(const DarknetGemmProblem& problem,
                    const std::vector<float>& weightsA,
                    const std::vector<float>& im2colB,
                    std::vector<float>& outputC);

void gemm_nt_xitao(const DarknetGemmProblem& problem,
                   const std::vector<float>& weightsA,
                   const std::vector<float>& im2colB,
                   std::vector<float>& outputC);

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs);

#endif
