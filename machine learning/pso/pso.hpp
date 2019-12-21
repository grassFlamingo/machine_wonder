#ifndef __H_PSO_ALG
#define __H_PSO_ALG

#include <stdlib.h>
#include <array>
#include <initializer_list>

inline float random_U01() { return float(rand() % 0x100) / (float)0x100; }

// additions
template <typename T, size_t N>
class matharray {
  typedef matharray<T, N> MType;

 protected:
  std::array<T, N> mItems;

 public:
  matharray() {}
  // support a = {1, 2, 3, ...}
  matharray(const std::initializer_list<T>& items) : mItems() {
    int maxlen = items.size() > N ? N : items.size();

    for (auto i = 0, it = items.begin(); i < maxlen; i++, it++) {
      this->mItems[i] = *it;
    }
  }

  matharray(const std::array<T, N>& items) : mItems(items) {}

  matharray(const matharray<T, N>& src) : mItems(src.mItems) {}

  matharray(const MType* src) : mItems(*src->mItems) {}

  void operator+=(MType& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] += b[i];
    }
  }

  void operator+=(T& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] += b;
    }
  }

  void operator-=(MType& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] -= b[i];
    }
  }

  void operator-=(T& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] -= b;
    }
  }

  void operator*=(MType& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] *= b[i];
    }
  }

  void operator*=(T& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] *= b;
    }
  }

  T& operator[](size_t i) { return this->mItems[i]; }

  T sum() {
    T ans = T(0);
    for (T i : this->mItems) {
      ans += i;
    }
    return ans;
  }

  void fill(MType& other) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] = other[i];
    }
  }

  void fill(const T& other) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] = other;
    }
  }

  // uniformly distribution [a,b]
  void random_uniform(float a = 0, float b = 1) {
    if (a > b) {
      auto t = a;
      a = b;
      b = t;
    }

    float scale = (b - a) / float(0x100);

    for (size_t i = 0; i < N; i++) {
      this->mItems[i] = float(rand() % 0x100) * scale + a;
    }
  }
};

template <typename T, size_t N>
class matharrayO : public matharray<T, N> {
  typedef matharrayO<T, N> MType;

 public:
  matharrayO(const matharray<T, N>& other) : matharray<T, N>(other) {}

  MType& operator+(MType& other) {
    *this += other;
    return *this;
  }

  MType& operator+(T& other) {
    for(T& i: this->mItems){
      i += other;
    }
    return *this;
  }

  MType& operator-(MType& other) {
    *this -= other;
    return *this;
  }

  MType& operator-(T& other) {
    for(T& i: this->mItems){
      i -= other;
    }
    return *this;
  }

  MType& operator*(MType& other) {
    for(size_t i = 0; i < N; i++){
      this->mItems[i] *= other[i];
    }
    return *this;
  }

  MType& operator*(T& other) {
    for(T& i: this->mItems){
      i *= other;
    }
    return *this;
  }
};

template <typename T, size_t N>
const matharrayO<T, N>& operator*(float a, const matharrayO<T, N>& me) {
  me *= a;
  return me;
}

// using is added in c++ 11
template <size_t N>
using matharray_f = matharray<float, N>;
template <size_t N>
using matharrayO_f = matharrayO<float, N>;

template <size_t N>
class Particles {
 public:
  Particles() {
    this->mPosition.random_uniform(-1, 1);
    this->mMyBest.fill(this->mPosition);
    this->mVelocity.fill(0.0f);
    this->mMinLoss = 1e10;
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
    matharrayO_f<N> _bv(this->mVelocity);
    matharrayO_f<N> _bp(this->mMyBest);
    matharrayO_f<N> _bg(globalbest);

    this->mVelocity += _bv * w;
    this->mVelocity += random_U01() * c1 * (_bp - this->mPosition);
    this->mVelocity += random_U01() * c2 * (_bg - this->mPosition);

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

  const size_t size = N;

 public:
  matharray_f<N> mPosition;
  matharray_f<N> mVelocity;
  matharray_f<N> mMyBest;
  float mMinLoss;
};

#endif  // __H_PSO_ALG