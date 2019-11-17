#ifndef __H_WE_MATRIX_H
#define __H_WE_MATRIX_H

#include <stdlib.h>

class Tuple {
 public:
  Tuple();
  ~Tuple();

};

class Matrix {
 public:
  Matrix(size_t width, size_t height);
  ~Matrix();

  size_t width();
  size_t height();
  bool is_shape(size_t w, size_t h);

  // const float& operator[](size_t row, size_t column);
  const Matrix& operator* (Matrix& other);


 private:
  float* mData;
  size_t mWidth;
  size_t mHeight;
};

/**
 * https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
 * Compute C = A B
 * C_{ik} = \sum_j A_{ij} B_{jk}
 * Please make sure that the last dimension of A equals to the first dimension of B
 * 
 * @param A(Matrix):
 * @param B(Matrix):
 * @param C(Matrix): 
 */
void matmul(Matrix& a, Matrix& b, Matrix& out);

#endif //__H_WE_MATRIX_H
