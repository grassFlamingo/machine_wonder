#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <array>

#include "pso.hpp"

#define DIMENSION_X 3
#define PARTICLES_N 8

matharray_f<DIMENSION_X> __loss_array = {0.3f, 0.5f, -0.1f};

float the_loss_function(matharray_f<DIMENSION_X> x) {
  auto t = matharrayO<float, DIMENSION_X>(x);
  t -= __loss_array;
  t = t * t;
  return t.sum();
}

int main(int argc, char const *argv[]) {
  srand(time(NULL));

  matharray_f<DIMENSION_X> globalBest;
  globalBest.random_uniform(-1, 1);

  Particles<DIMENSION_X> partList[PARTICLES_N];
  float gloss = the_loss_function(globalBest);

  printf("Loss of initial global is %f \n", gloss);

  for (int i = 0; i < PARTICLES_N; i++) {
    auto pi = partList[i];
    float tloss = the_loss_function(pi.mPosition);
    pi.mMinLoss = tloss;
    if (tloss < gloss) {
      pi.mMyBest = globalBest;
    }
    pi.mVelocity.random_uniform(-2.0, 2.0);
  }

  for (int i = 0; i < 200; i++) {
    // loop
    for (auto pi : partList) {
      pi.update_position(globalBest, 0.8, 0.5, 0.2);
      float lpi = the_loss_function(pi.mPosition);
      pi.set_my_best(lpi);
      if (lpi < gloss) {
        gloss = lpi;
        globalBest.fill(pi.mPosition);
      }
    }
  }

  printf("Loss of final global is %f \n", gloss);

  return 0;
}

/**
 * 523MB (523,239,424 bytes) Microsoft WIndows Recovery Environment (System)
 * 105MB EFI System Fat 32
 *
 * 17 MB (16,777.216 bytes) Microsoft Reserved
 * 107GB NTFS
 * 664MB (663,748,608 bytes) Microsoft Recovery Environment (System, No
 * Automount) NTFS
 *
 *
 *
 */
