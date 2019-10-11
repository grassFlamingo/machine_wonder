#include "theColumn.h"

///////////////////////////////////////
// ColumnType
template <typename T>
ColumnItem::ColumnItem(ColumnType type, T value){
  this->mType = type;
  *this->mItem = *((unsigned char*) &value);
}

ColumnItem::ColumnItem() : mType(ColumnType::cint64) {
  memset(&this->mItem, 0, COLUMN_ITEM_BITS_MAX);
}

ColumnItem::~ColumnItem() {}

ColumnType ColumnItem::type() { return this->mType; }

/**
 * read `size` to pointer `p`
 */
void ColumnItem::get_value(void* p, u_int8_t size) {
  if (size > COLUMN_ITEM_BITS_MAX) {
    return;
  }
  unsigned char* vp = (unsigned char*)p;
  for (int i = 0; i < size; i++) {
    vp[i] = this->mItem[i];
  }
}

int32_t ColumnItem::int32_value() { return *((int32_t*)this->mItem); }

int8_t ColumnItem::int8_value(){ return (int8_t)this->mItem[0]; }

///////////////////////////////////////
// This belows are BaseColumn
BaseColumn::BaseColumn() {}

BaseColumn::~BaseColumn() {}

int BaseColumn::length() { return this->mLength; }

ColumnType BaseColumn::type() { return this->mType; }

///////////////////////////////////////
// This belows are Int32Column
Int32Column::Int32Column() {}

Int32Column::~Int32Column() {}

ColumnItem Int32Column::get_value(int index){
  return ColumnItem(ColumnType::cint32, this->mRows[index]);
}

///////////////////////////////////////
// This belows are Int8Column
Int8Column::Int8Column() {}

Int8Column::~Int8Column() {}

ColumnItem Int8Column::get_value(int index){
  return ColumnItem(ColumnType::cint8, this->mRows[index]);
}
