#include "../lib/matharray.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * normal distribution
 */
float rand_normal(float mu, float sigma) {
  float r, v1, v2, fac;

  r = 2;
  while (r >= 1) {
    v1 = 2 * (random_U01() - 1);
    v2 = 2 * (random_U01() - 1);
    r = v1 * v1 + v2 * v2;
  }
  fac = sqrt(-2 * log(r) / r);

  return sigma * v2 * fac + mu;
}

float distance_ecu(float x1, float y1, float x2, float y2) {
  float dx = x1 - x2;
  float dy = y1 - y2;
  return sqrtf32(dx * dx + dy * dy);
}

#define dataPoints 256
#define centers 4

// debug needed
int main(int argc, char const* argv[]) {
  srand(time(NULL));

  matharray<float, dataPoints> x;
  matharray<float, dataPoints> y;
  for (int i = 0; i < centers; i++) {
    float cx = random_U01() * (i + 1) * 2.0;
    float cy = random_U01() * (i + 1) * 2.0;
    printf("[center %d] (%.4f, %.4f)\n", i, cx, cy);
    float* tx = &x[i * dataPoints / centers];
    float* ty = &y[i * dataPoints / centers];
    for (int j = 0; j < dataPoints / centers; j++) {
      tx[j] = rand_normal(cx, 0.5);
      ty[j] = rand_normal(cy, 0.5);
    }
  }


  matharray<float, centers> cx;
  matharray<float, centers> cy;

  // init, random center
  for (int i = 0; i < centers; i++) {
    int t = rand() % dataPoints;
    cx[i] = x[t];
    cy[i] = y[t];
  }
  matharray<int, dataPoints> assignments;

  // clustering
  for (int i = 0; i < 100; i++) {
    // compute all distance to centers
    for (int j = 0; j < dataPoints; j++) {
      float xj = x[j], yj = y[j];
      float mintdc = INFINITY;
      int minc = 0;
      for (int c = 0; c < centers; c++) {
        float td = distance_ecu(xj, yj, cx[c], cy[c]);
        if (td < mintdc) {
          mintdc = td;
          minc = c;
        }
      }
      assignments[j] = minc;
    }
    // update center
    cx.fill(0);
    cy.fill(0);
    matharray<int, centers> countc;
    for (int j = 0; j < dataPoints; j++) {
      int ass = assignments[j];
      cx[ass] += x[j];
      cy[ass] += y[j];
      countc[ass] += 1;
    }
    cx /= countc.cast<float>();
    cy /= countc.cast<float>();

    if(i % 10 != 0){
      continue;
    }
    for (int i = 0; i < centers; i++) {
      printf("[%d] (%.4f, %.4f)\n", i, cx[i], cy[i]);
    }
    puts("----------");
  }

  return 0;
}
