#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <array>

#include "pso.hpp"

#define PARTICLES_N 16
#define DIMENSION_X 3

class GradCache {
 public:
  float mLoss;
  matharray_f<DIMENSION_X> mGrad;
  matharray_f<DIMENSION_X> mX;

  GradCache(float a = -1.5, float b = 1.5) {
    this->mX.random_uniform(a, b);
    this->mLoss = 0.0f;
  }

  GradCache(const matharray_f<DIMENSION_X>& other) {
    this->mX = other;
    this->mLoss = 0.0f;
  }
};

// #define X_SQUARE
#define XX_SIN_X

#ifdef X_SQUARE
matharray_f<DIMENSION_X> __loss_array = {0.1f, 1.2f, -0.9f};

void the_loss_function(GradCache& gcache) {
  auto t = gcache.mX - __loss_array;
  gcache.mGrad = t * 2;
  t *= t;
  gcache.mLoss = t.sum();
}

#elif defined(XX_SIN_X)
matharray_f<DIMENSION_X> __loss_array = {-0.48f, -0.48f, -0.48f};

void the_loss_function(GradCache& gcache) {
  auto sq = (gcache.mX * gcache.mX) / 3.0f;
  auto s3 = gcache.mX * 3.0f;
  auto ss = matharray_f<DIMENSION_X>::sin(s3);
  gcache.mLoss = sq.sum() + ss.sum() + 1;
  gcache.mGrad =
      gcache.mX * (2.0 / 3.0) + matharray_f<DIMENSION_X>::cos(s3) * 3.0;
}
#else
#error("loss function undefined!")
#endif

void print_math_array(matharray_f<DIMENSION_X> x, bool newline = true) {
  printf("[");
  for (float j : x) {
    printf(" %f", j);
  }
  if (newline) {
    printf(" ]\n");
  } else {
    printf(" ]");
  }
}

int main(int argc, char const* argv[]) {
  srand(time(NULL));

  size_t nIteration = 100;
  float lr = 2e-2;
  if (argc >= 2) {
    sscanf(argv[1], "%lu", &nIteration);
  }
  if (argc >= 3) {
    sscanf(argv[2], "%f", &lr);
  }

  GradCache gcache(-1.5, 1.5);

  for (size_t i = 0; i < nIteration; i++) {
    the_loss_function(gcache);

    if (i % 1 == 0) {
      printf("epho %03lu gloss %.5f ", i, gcache.mLoss);
      print_math_array(gcache.mX, true);
    }

    gcache.mX -= gcache.mGrad * lr;
  }

  printf("real value ");
  print_math_array(__loss_array, true);
  printf("Loss of final global is %f \n", gcache.mLoss);

  return 0;
}
