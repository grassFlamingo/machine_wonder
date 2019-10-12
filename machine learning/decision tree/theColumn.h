#ifndef __ML_DATA_COLUMN_H
#define __ML_DATA_COLUMN_H

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "utils.h"

enum ColumnType {
  none,
  int32,
  int64,
  int8,
  float32,
  float64,
};

class NoneColumn {
 protected:
  ColumnType mType;

 public:
  NoneColumn();
  ~NoneColumn();

  void append(int item);
  void set_value(int index, int item);
  int& get_value(int index);

  int& operator[] (int index);

  size_t length();
  ColumnType type();
};

template <class C, ColumnType M>
class TemplateColumn : public NoneColumn {
 private:
  std::vector<C> mRows;

 public:
  TemplateColumn() { this->mType = M; }
  ~TemplateColumn() {}

  void append(C item) { this->mRows.push_back(item); }
  void set_value(int index, C item) {
    if (index < this->mRows.size()) {
      this->mRows[index] = item;
    }
    throw IndexOutOfBoundError();
  }
  C& get_value(int index) {
    if (index < this->mRows.size()) {
      return this->mRows[index];
    }
    throw IndexOutOfBoundError();
  }
  C& operator[](int index) {
    if (index < this->mRows.size()) {
      return this->mRows[index];
    }
    throw IndexOutOfBoundError();
  }

  size_t length() { return this->mRows.size(); }
};

typedef TemplateColumn<int32_t, ColumnType::int32> Int32Column;
typedef TemplateColumn<int8_t, ColumnType::int8> Int8Column;
typedef TemplateColumn<float, ColumnType::float32> Float32Column;
typedef TemplateColumn<double, ColumnType::float64> Float64Column;

#endif  //__ML_DATA_COLUMN_H