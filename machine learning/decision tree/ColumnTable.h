#ifndef __ML_COLUMN_TABLE_H
#define __ML_COLUMN_TABLE_H

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include "Utils.h"

enum ColumnType {
  none,
  int8,
  int32,
  float32,
  float64,
};

class NoneColumn {
 protected:
  ColumnType mType;
  std::string mName;
 public:
  NoneColumn();
  NoneColumn(std::string name);
  virtual ~NoneColumn();

  virtual void set_value(size_t index, int8_t value) = 0;
  virtual void set_value(size_t index, int32_t value) = 0;
  virtual void set_value(size_t index, float value) = 0;
  virtual void set_value(size_t index, double value) = 0;
  virtual void set_value(size_t index, char* value) = 0;
  virtual void set_value(size_t index, std::string value) = 0;

  virtual void append(int8_t value) = 0;
  virtual void append(int32_t value) = 0;
  virtual void append(float value) = 0;
  virtual void append(double value) = 0;
  virtual void append(char* value) = 0;
  virtual void append(std::string value) = 0;

  virtual int8_t get_int8(size_t index) = 0;
  virtual int32_t get_int32(size_t index) = 0;
  virtual float get_float32(size_t index) = 0;
  virtual double get_float64(size_t index) = 0;

  virtual double mean() = 0;
  virtual double sum() = 0;

  virtual size_t length();

  /**
   * Count different values
   */
  virtual std::vector<int> value_counts(double epsilon=0) = 0;

  virtual NoneColumn* subcolumn(int8_t value) = 0;
  virtual NoneColumn* subcolumn(int32_t value) = 0;
  virtual NoneColumn* subcolumn(float value, double epsilon) = 0;
  virtual NoneColumn* subcolumn(double value, double epsilon) = 0;
  

  ColumnType type();
  std::string name();
  const char* name_cstr();
};

class Float32Column: public NoneColumn{
 private:
  std::vector<float> mRows;
 public:
  Float32Column();
  Float32Column(std::string name);
  virtual ~Float32Column();

  virtual void set_value(size_t index, int8_t value);
  virtual void set_value(size_t index, int32_t value);
  virtual void set_value(size_t index, float value);
  virtual void set_value(size_t index, double value);
  virtual void set_value(size_t index, char* value);
  virtual void set_value(size_t index, std::string value);

  virtual void append(int8_t value);
  virtual void append(int32_t value);
  virtual void append(float value);
  virtual void append(double value);
  virtual void append(char* value);
  virtual void append(std::string value);

  virtual int8_t get_int8(size_t index);
  virtual int32_t get_int32(size_t index);
  virtual float get_float32(size_t index);
  virtual double get_float64(size_t index);

  /**
   * Count different values
   */
  virtual std::vector<int> value_counts(double epsilon=1e-8);

  virtual NoneColumn* subcolumn(int8_t value);
  virtual NoneColumn* subcolumn(int32_t value);
  virtual NoneColumn* subcolumn(float value, double epsilon);
  virtual NoneColumn* subcolumn(double value, double epsilon);

  virtual double mean();
  virtual double sum();

  virtual size_t length();
};

class Int32Column: public NoneColumn{
 private:
  std::vector<int32_t> mRows;
 public:
  Int32Column();
  Int32Column(std::string name);
  virtual ~Int32Column();

  virtual void set_value(size_t index, int8_t value);
  virtual void set_value(size_t index, int32_t value);
  virtual void set_value(size_t index, float value);
  virtual void set_value(size_t index, double value);
  virtual void set_value(size_t index, char* value);
  virtual void set_value(size_t index, std::string value);

  virtual void append(int8_t value);
  virtual void append(int32_t value);
  virtual void append(float value);
  virtual void append(double value);
  virtual void append(char* value);
  virtual void append(std::string value);

  virtual int8_t get_int8(size_t index);
  virtual int32_t get_int32(size_t index);
  virtual float get_float32(size_t index);
  virtual double get_float64(size_t index);

  /**
   * Count different values
   */
  virtual std::vector<int> value_counts(double epsilon=0);

  virtual NoneColumn* subcolumn(int8_t value);
  virtual NoneColumn* subcolumn(int32_t value);
  virtual NoneColumn* subcolumn(float value, double epsilon);
  virtual NoneColumn* subcolumn(double value, double epsilon);

  virtual double mean();
  virtual double sum();

  virtual size_t length();
};

#endif  //__ML_COLUMN_TABLE_H