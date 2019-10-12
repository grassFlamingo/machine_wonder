#include "theColumn.h"

///////////////////////////////////////
// This belows are NoneColumn
NoneColumn::NoneColumn() {}

NoneColumn::~NoneColumn() {}

void NoneColumn::append(int item) {
  throw NotImplementedError();
}
void NoneColumn::set_value(int index, int item) {
  throw NotImplementedError();
}
int& NoneColumn::get_value(int index) {
  throw NotImplementedError();
}
int& NoneColumn::operator[](int index) {
  throw NotImplementedError();
}

size_t NoneColumn::length() { return 0; }

ColumnType NoneColumn::type() { return this->mType; }
