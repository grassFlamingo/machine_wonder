#ifndef _MATH_ARRAY_H
#define _MATH_ARRAY_H

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <initializer_list>

inline float random_U01() { return float(rand() % 0xf00) / (float)0xf00; }

// additions
template <typename T, size_t N>
class matharray {
  typedef matharray<T, N> MType;

 protected:
  T mItems[N];

 public:
  matharray() { memset(mItems, 0, sizeof(T) * N); }
  // support a = {1, 2, 3, ...}
  matharray(const std::initializer_list<T>& items) {
    int maxlen = items.size() > N ? N : items.size();

    for (auto i = 0, it = items.begin(); i < maxlen; i++, it++) {
      this->mItems[i] = *it;
    }
  }

  matharray(const MType* src) { memcpy(this->mItems, src, N * sizeof(T)); }

  matharray(const MType& src) : matharray(&src) {}

  template <typename C>
  matharray<C, N> cast() {
    matharray<C, N> ans;
    for (size_t i = 0; i < N; i++) {
      ans[i] = (C)this->mItems[i];
    }
    return ans;
  }

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

  MType operator+(const MType& b) { return MType(this) += b; }
  MType operator+(T& b) { return MType(this) += b; }

  const MType& operator-=(const MType& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] -= b.mItems[i];
    }
    return *this;
  }

  const MType& operator-=(const T& b) {
    for (size_t i = 0; i < N; i++) {
      this->mItems[i] -= b;
    }
    return *this;
  }

  const MType operator-(const MType& b) {
    MType ans(this);
    ans -= b;
    return ans;
  }
  const MType& operator-(T& b) { return MType(this) -= b; }

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

  MType operator*(const MType& b) { return MType(this) *= b; }
  MType operator*(const T& b) { return MType(this) *= b; }

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

  MType operator/(const MType& b) { return MType(this) /= b; }
  MType operator/(const T& b) { return MType(this) /= b; }

  T& operator[](size_t i) { return this->mItems[(i + N) % N]; }

  inline size_t lenght() const { return N; }

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
  T* begin() { return this->mItems; }

  T* end() { return this->mItems + N; }

  T sum() {
    T ans = T(0);
    for (T i : this->mItems) {
      ans += i;
    }
    return ans;
  }

  void shuffle() {
    for (int i = 0; i < (int)N - 1; i++) {
      T t = mItems[i];
      int j = rand() % (N - i) + i;
      mItems[i] = mItems[j];
      mItems[j] = t;
    }
  }

  void shuffle(matharray<int, N>& index) {
    for (int i = 0; i < (int)N; i++) {
      T t = mItems[i];
      int j = index[i];
      if (j >= (int)N) {
        continue;
      }
      mItems[i] = mItems[j];
      mItems[j] = t;
    }
  }

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