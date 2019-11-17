#include "matrix.h"

Matrix::Matrix(size_t width, size_t height) {
  this->mWidth = width;
  this->mHeight = height;
  this->mData = new float[width * height];
}

Matrix::~Matrix() {
  if (this->mData != NULL) {
    delete[] this->mData;
  }
}

size_t Matrix::width() { return this->mWidth; }

size_t Matrix::height() { return this->mHeight; }

bool Matrix::is_shape(size_t w, size_t h) {
  return (w == this->mWidth) & (h == this->mHeight);
}

/**
 * https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
 * Compute C = A B
 * C_{ik} = \sum_j A_{ij} B_{jk}
 * Please make sure that the last dimension of A equals to the first dimension
 * of B
 *
 * @param A(Matrix): A_{ij}
 * @param B(Matrix): B_{jk}
 * @param C(Matrix): C_{ik}
 */
void matmul(Matrix& A, Matrix& B, Matrix& C) {
  // TODO:
  // if n = 1, set c11 ← a11 × b11 (or multiply a small block matrix)
  if (A.is_shape(1,1) && B.is_shape(1,1)) {
    
  } 
  // Otherwise, allocate space for a new matrix T of shape n × n, then

}

/**

Procedure multiply(C, A, B):
  Base case: if n = 1, set c11 ← a11 × b11 (or multiply a small block matrix).
  Otherwise, allocate space for a new matrix T of shape n × n, then:
    Partition A into A11, A12, A21, A22.
    Partition B into B11, B12, B21, B22.
    Partition C into C11, C12, C21, C22.
    Partition T into T11, T12, T21, T22.
    Parallel execution:
      Fork multiply(C11, A11, B11).
      Fork multiply(C12, A11, B12).
      Fork multiply(C21, A21, B11).
      Fork multiply(C22, A21, B12).
      Fork multiply(T11, A12, B21).
      Fork multiply(T12, A12, B22).
      Fork multiply(T21, A22, B21).
      Fork multiply(T22, A22, B22).
    Join (wait for parallel forks to complete).
    add(C, T).
    Deallocate T.

Procedure add(C, T) adds T into C, element-wise:
  Base case: if n = 1, set c11 ← c11 + t11 (or do a short loop, perhaps unrolled).
  Otherwise:
    Partition C into C11, C12, C21, C22.
    Partition T into T11, T12, T21, T22.
    In parallel:
      Fork add(C11, T11).
      Fork add(C12, T12).
      Fork add(C21, T21).
      Fork add(C22, T22).
    Join.
*/