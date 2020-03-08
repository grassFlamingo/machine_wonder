#ifndef __H_PSO_ALG
#define __H_PSO_ALG

#include "../lib/matharray.hpp"
#include <stdlib.h>

// using is added in c++ 11
template <size_t N>
using matharray_f = matharray<float, N>;

template <size_t N>
class Particles {
 public:
  /**
   * mPosition <- U(low, up)
   * mVelocity <- U(low, up)
   * @param low:
   * @param up;
   */
  Particles(float low = -1.0f, float up = 1.0f) {
    this->mPosition.random_uniform(low, up);
    this->mMyBest.fill(this->mPosition);
    this->mVelocity.random_uniform(low, up);
    this->mMinLoss = 1e+10;
  }

  Particles(const Particles<N>& other) {
    this->mPosition = other.mPosition;
    this->mMyBest = other.mMyBest;
    this->mVelocity = other.mVelocity;
    this->mMinLoss = other.mMinLoss;
  }

  ~Particles() {}
  /**
   * Update velocity and position following:
   * v(t+1) = w v(t) + r1 c1 (p(t) - x(t)) + r2 c2 (g(t) - x(t))
   * x(t+1) = x(t) + v(t+1)
   *
   * - x(t), x(t+1): current position, next position
   * - v(t). v(t+1): current speed, next speed
   * - r1, r2 is two random number in U(0,1)
   * - w, c1, c2 come from param
   *
   * @param globalbest: g(t)
   * @param w: w
   * @param c1: c1
   * @param c2: c2
   */
  void update_position(matharray_f<N>& globalbest, float w, float c1,
                       float c2) {
    matharray_f<N> _bv(this->mVelocity);
    matharray_f<N> _bp = this->mMyBest - this->mPosition;
    matharray_f<N> _bg = globalbest - this->mPosition;

    this->mVelocity =
        _bv * w + _bp * (c1 * random_U01()) + _bg * (c2 * random_U01());

    this->mPosition += this->mVelocity;
  }

  /**
   * This function update the mByBest
   * when `loss` is less than my history mininum loss.
   * @param loss: the loss come from loss function
   */
  void set_my_best(float loss) {
    if (loss > this->mMinLoss) {
      return;
    }
    this->mMinLoss = loss;
    this->mMyBest.fill(this->mPosition);
  }

  void operator=(const Particles<N>& other) {
    this->mPosition = other.mPosition;
    this->mMyBest = other.mMyBest;
    this->mVelocity = other.mVelocity;
    this->mMinLoss = other.mMinLoss;
  }

  const size_t size = N;

 public:
  matharray_f<N> mPosition;
  matharray_f<N> mVelocity;
  matharray_f<N> mMyBest;
  float mMinLoss;
};

#endif  // __H_PSO_ALG