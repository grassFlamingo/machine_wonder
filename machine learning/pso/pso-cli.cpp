#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <array>

#include "pso.hpp"

#define DIMENSION_X 3

// #define X_SQUARE
#define XX_SIN_X

#ifdef X_SQUARE
#define PARTICLES_N 16
matharray_f<DIMENSION_X> __loss_array = {0.1f, 1.2f, -0.9f};

float the_loss_function(matharray_f<DIMENSION_X> x) {
  auto t = x -  __loss_array;
  t *= t;
  return t.sum();
}
#elif defined(XX_SIN_X)
#define PARTICLES_N 64
matharray_f<DIMENSION_X> __loss_array = {-0.48f, -0.48f, -0.48f};

float the_loss_function(matharray_f<DIMENSION_X> x){
  auto sq = (x * x) / 3.0f;
  auto ss = matharray_f<DIMENSION_X>::sin(x * 3.0f);
  return sq.sum() + ss.sum() + 1;
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

/**
 * pso-cli iteration
 * - iteration: default 100
 */
int main(int argc, char const *argv[]) {
  srand(time(NULL));

  size_t nIteration = 100;
  if(argc >= 2){
    sscanf(argv[1], "%lu", &nIteration);
  }

  matharray_f<DIMENSION_X> globalBest;
  globalBest.random_uniform(-1.2, 1.2);
  float gloss = the_loss_function(globalBest);

  Particles<DIMENSION_X> partList[PARTICLES_N];

  for(size_t i = 0; i < PARTICLES_N; i++){
    partList[i] = Particles<DIMENSION_X>(-1.2f, 1.2f);
  }

  printf("Loss of initial global is %f \n", gloss);

  for (auto& pi: partList) {
    float tloss = the_loss_function(pi.mPosition);
    pi.mMinLoss = tloss;
    if (tloss < gloss) {
      gloss = tloss;
      globalBest.fill(pi.mPosition);
    }
  }

  for (size_t i = 0; i < nIteration; i++) {
    // loop
    // print_math_array(partList[0].mPosition);
    for (auto& pi : partList) {
      #ifdef X_SQUARE
      pi.update_position(globalBest, 0.8, 1.2, 1.0);
      #else
      pi.update_position(globalBest, 1.0, 1.2, 1.0);
      #endif
      float lpi = the_loss_function(pi.mPosition);
      pi.set_my_best(lpi);
      if (lpi < gloss) {
        gloss = lpi;
        globalBest.fill(pi.mPosition);
      }
    }
    // print_math_array(partList[0].mPosition);
    if (i % 1 == 0) {
      printf("epho %03lu gloss %.5f ", i, gloss);
      print_math_array(globalBest, true);
    }
  }

  printf("real value ");
  print_math_array(__loss_array, true);
  printf("Loss of final global is %f \n", gloss);

  return 0;
}
