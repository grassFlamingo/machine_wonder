#ifndef _MATH_ARRAY_H
#define _MATH_ARRAY_H

#include <math.h>
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

  const MType& operator+=(const MType& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] += b.mItems[i];
    }
    return *this;
  }

  const MType& operator+=(T& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] += b;
    }
    return *this;
  }

  MType operator+(const MType& b) { return MType(this->mItems) += b; }
  MType operator+(T& b) { return MType(this->mItems) += b; }

  const MType& operator-=(const MType& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] -= b.mItems[i];
    }
    return *this;
  }

  const MType& operator-=(T& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] -= b;
    }
    return *this;
  }

  const MType operator-(const MType& b) {
    MType ans(this->mItems);
    ans -= b;
    return ans;
  }
  const MType& operator-(T& b) { return MType(this->mItems) -= b; }

  MType& operator*=(const MType& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] *= b.mItems[i];
    }
    return *this;
  }

  MType& operator*=(const T& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] *= b;
    }
    return *this;
  }

  MType operator*(const MType& b) { return MType(this->mItems) *= b; }
  MType operator*(const T& b) { return MType(this->mItems) *= b; }

  MType& operator/=(const MType& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] /= b.mItems[i];
    }
    return *this;
  }

  MType& operator/=(const T& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] /= b;
    }
    return *this;
  }

  MType operator/(const MType& b) { return MType(this->mItems) /= b; }
  MType operator/(const T& b) { return MType(this->mItems) /= b; }

  T operator[](size_t i) { return this->mItems[i]; }

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

  // iterator
  T* begin() { return this->mItems.begin(); }

  T* end() { return this->mItems.end(); }

  // static
  static matharray<T, N> sin(const matharray<T, N>& x) {
    matharray<T, N> ans(x);
    for (auto& i : ans) {
      i = std::sin(i);
    }
    return ans;
  }

  // static
  static matharray<T, N> cos(const matharray<T, N>& x) {
    matharray<T, N> ans(x);
    for (auto& i : ans) {
      i = std::cos(i);
    }
    return ans;
  }
};

#endif  //_MATH_ARRAY_H