//
// OpenMP benchmark for measuring effective memory bandwidth for strided array access
//
// Author: Karl Rupp,  me@karlrupp.net
// License: MIT/X11 license, see file LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cstdlib>

#include "benchmark-utils.hpp"

typedef double       NumericT;

int kernel_func(NumericT *x, NumericT const *y, NumericT const *z, int stride, int N)
{
  if (stride == 1)
  {
    #pragma omp parallel for
    for (int i=0; i<N; ++i)
      x[i] = y[i] + z[i];
  }
  else
  {
    #pragma omp parallel for
    for (int i=0; i<N; ++i)
      x[i*stride] = y[i*stride] + z[i*stride];
  }
}

int main(int argc, char **argv)
{

  // slightly larger on CPU than on GPU so that arrays don't fit in cache
  std::size_t N = 20 * 1000 * 1000;

  NumericT *x, *y, *z;

  if (posix_memalign((void**)&x, 64, 32*N*sizeof(NumericT)))
    throw std::runtime_error("Failed to allocate x");
  if (posix_memalign((void**)&y, 64, 32*N*sizeof(NumericT)))
    throw std::runtime_error("Failed to allocate y");
  if (posix_memalign((void**)&z, 64, 32*N*sizeof(NumericT)))
    throw std::runtime_error("Failed to allocate z");

  #pragma omp parallel for
  for (std::size_t i=0; i<32*N; ++i)
  {
    x[i] = 1.0;
    y[i] = 2.0;
    z[i] = 3.0;
  }

  // warmup:
  kernel_func(&x[0], &y[0], &z[0], 1, N);

  // Benchmark runs
  Timer timer;
  std::cout << "# stride     time       GB/sec" << std::endl;
  for (std::size_t stride = 1; stride <= 32 ; ++stride)
  {
    timer.start();

    // repeat calculation several times, then average
    for (std::size_t num_runs = 0; num_runs < 20; ++num_runs)
    {
      kernel_func(&x[0], &y[0], &z[0], stride, N);
    }
    double exec_time = timer.get();

    std::cout << "   " << stride << "        " << exec_time << "        " << 20 * 3.0 * sizeof(NumericT) * N / exec_time * 1e-9 << std::endl;
  }

  return EXIT_SUCCESS;
}
