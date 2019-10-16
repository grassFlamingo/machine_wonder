#include "ColumnTable.h"

///////////////////////////////////////
// This belows are NoneColumn
NoneColumn::NoneColumn() { this->mType = ColumnType::none; }
NoneColumn::NoneColumn(std::string name) {
  this->mName = name;
  this->mType = ColumnType::none;
}
NoneColumn::~NoneColumn() {}

ColumnType NoneColumn::type() { return this->mType; }

size_t NoneColumn::length() { return 0u; }

std::string NoneColumn::name() { return this->mName; }

const char* NoneColumn::name_cstr() { return this->mName.c_str(); }


/* Float32Column */

Float32Column::Float32Column() { this->mType = ColumnType::float32; }

Float32Column::Float32Column(std::string name) {
  this->mName = name;
  this->mType = ColumnType::float32;
}
Float32Column::~Float32Column() {}

void Float32Column::set_value(size_t index, int8_t value) {
  this->set_value(index, (float)value);
}
void Float32Column::set_value(size_t index, int32_t value) {
  this->set_value(index, (float)value);
}
void Float32Column::set_value(size_t index, float value) {
  if (index >= this->mRows.size()) {
    throw IndexOutOfBoundError();
  }
  this->mRows[index] = value;
}

void Float32Column::set_value(size_t index, double value) {
  this->set_value(index, (float)value);
}
void Float32Column::set_value(size_t index, char* value) {
  this->set_value(index, std::stof(value));
}
void Float32Column::set_value(size_t index, std::string value) {
  this->set_value(index, std::stof(value));
}

void Float32Column::append(int8_t value) { this->append((float)value); }
void Float32Column::append(int32_t value) { this->append((float)value); }
void Float32Column::append(float value) { this->mRows.push_back(value); }
void Float32Column::append(double value) { this->append((float)value); }
void Float32Column::append(char* value) { this->append(std::stof(value)); }
void Float32Column::append(std::string value) {
  this->append(std::stof(value));
}

int8_t Float32Column::get_int8(size_t index) {
  return (int8_t)this->get_float32(index);
}

int32_t Float32Column::get_int32(size_t index) {
  return (int32_t)this->get_float32(index);
}
float Float32Column::get_float32(size_t index) {
  if (index >= this->mRows.size()) {
    throw IndexOutOfBoundError();
  }
  return this->mRows[index];
}
double Float32Column::get_float64(size_t index) {
  return (double)this->get_float32(index);
}

std::vector<int> Float32Column::value_counts(double epsilon){
  throw NotImplementedError();
}

NoneColumn* Float32Column::subcolumn(int8_t value){
  throw NotImplementedError();
}

NoneColumn* Float32Column::subcolumn(int32_t value){
  throw NotImplementedError();
}

NoneColumn* Float32Column::subcolumn(float value, double epsilon){
  throw NotImplementedError();
}

NoneColumn* Float32Column::subcolumn(double value, double epsilon){
  throw NotImplementedError();
}


double Float32Column::mean() {
  double tsum = 0;
  for (float c : this->mRows) {
    tsum += c;
  }
  return tsum / (float)this->mRows.size();
}

double Float32Column::sum() {
  double tsum = 0;
  for (float c : this->mRows) {
    tsum += c;
  }
  return tsum;
}

size_t Float32Column::length() { return this->mRows.size(); }

/* Int 32 COlumn */

Int32Column::Int32Column() { this->mType = ColumnType::float32; }

Int32Column::Int32Column(std::string name) {
  this->mName = name;
  this->mType = ColumnType::int32;
}
Int32Column::~Int32Column() {}

void Int32Column::set_value(size_t index, int8_t value) {
  this->set_value(index, (int32_t)value);
}
void Int32Column::set_value(size_t index, int32_t value) {
  if (index >= this->mRows.size()) {
    throw IndexOutOfBoundError();
  }
  this->mRows[index] = value;
}
void Int32Column::set_value(size_t index, float value) {
  this->set_value(index, (int32_t)value);
}

void Int32Column::set_value(size_t index, double value) {
  this->set_value(index, (int32_t)value);
}
void Int32Column::set_value(size_t index, char* value) {
  this->set_value(index, std::stoi(value));
}
void Int32Column::set_value(size_t index, std::string value) {
  this->set_value(index, std::stoi(value));
}

void Int32Column::append(int8_t value) { this->append((int32_t)value); }
void Int32Column::append(int32_t value) { this->mRows.push_back(value); }
void Int32Column::append(float value) { this->append((int32_t)value); }
void Int32Column::append(double value) { this->append((int32_t)value); }
void Int32Column::append(char* value) { this->append(std::stoi(value)); }
void Int32Column::append(std::string value) { this->append(std::stoi(value)); }

int8_t Int32Column::get_int8(size_t index) {
  return (int8_t)this->get_int32(index);
}

int32_t Int32Column::get_int32(size_t index) {
  if (index >= this->mRows.size()) {
    throw IndexOutOfBoundError();
  }
  return this->mRows[index];
}
float Int32Column::get_float32(size_t index) {
  return (int32_t)this->get_int32(index);
}
double Int32Column::get_float64(size_t index) {
  return (double)this->get_int32(index);
}

std::vector<int> Int32Column::value_counts(double epsilon){
  std::map<int, int> valueMcount;
  for(int32_t i: this->mRows){
    valueMcount[i] += 1;
  }
  std::vector<int> counts(valueMcount.size());
  int j = 0;
  for(auto i: valueMcount){
    counts[j++] = i.second;
  }
  return counts;
}

NoneColumn* Int32Column::subcolumn(int8_t value){
  throw NotImplementedError();
}

NoneColumn* Int32Column::subcolumn(int32_t value){
  throw NotImplementedError();
}

NoneColumn* Int32Column::subcolumn(float value, double epsilon){
  throw NotImplementedError();
}

NoneColumn* Int32Column::subcolumn(double value, double epsilon){
  throw NotImplementedError();
}


double Int32Column::mean() {
  double tsum = 0;
  for (float c : this->mRows) {
    tsum += c;
  }
  return tsum / (double)this->mRows.size();
}

double Int32Column::sum() {
  double tsum = 0;
  for (float c : this->mRows) {
    tsum += c;
  }
  return tsum;
}

size_t Int32Column::length() { return this->mRows.size(); }
