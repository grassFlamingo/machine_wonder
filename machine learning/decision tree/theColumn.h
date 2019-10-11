#ifndef __ML_DATA_COLUMN_H
#define __ML_DATA_COLUMN_H

#include <stdlib.h>
#include <memory.h>

enum ColumnType{
  cbase,
  cint32,
  cint64,
  cint8,
};

//64 bits
#define COLUMN_ITEM_BITS_MAX 8

class ColumnItem{
 private:
  unsigned char mItem[COLUMN_ITEM_BITS_MAX]; //64 bits
  ColumnType mType;

 public:
  template <typename T>
  ColumnItem(ColumnType type, T value);

  ColumnItem();
  ~ColumnItem();

  ColumnType type();
  /**
   * read `size` to pointer `p`
   */
  void get_value(void* p, u_int8_t size);
  int32_t int32_value();
  int8_t int8_value();
};

class BaseColumn {
 private:
  ColumnType mType;
 protected:
  int mLength;
 public:
  BaseColumn();
  ~BaseColumn();

  virtual bool set_value(int index, ColumnItem &item);

  template <typename T>
  void set_value(int index, T item);

  virtual ColumnItem get_value(int index);

  int length();
  ColumnType type();
};

class Int32Column : BaseColumn {
 private:
  int32_t* mRows;

 public:
  Int32Column();
  ~Int32Column();

  ColumnItem get_value(int index);
};

class Int8Column{
private:
  int8_t* mRows;
public:
  Int8Column();
  ~Int8Column();

  ColumnItem get_value(int index);
};

#endif  //__ML_DATA_COLUMN_H