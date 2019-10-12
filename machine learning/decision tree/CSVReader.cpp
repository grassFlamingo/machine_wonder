#include "CSVReader.h"

/**
 * @param filename
 * @usehead(bool) whether thereis a head line in CSV file
 * @separator(char) the separator of each field, usually `,`
 */
CSVReader::CSVReader(const char* filename, bool usehead, char separator) {
  this->mSeparator = separator;
  this->mFile.open(filename, std::ios::in);
  if (!this->mFile) {
    throw FileOpenError();
  }
  this->mColumnNames = this->read_line();
  if (!usehead) {
    char col[32];
    int size = this->mColumnNames.size();
    for (int i = 0; i < size; i++) {
      sprintf(col, "col-%02d", i);
      this->mColumnNames[i] = col;
    }
    this->mFile.seekg(0, std::ios::beg);
  }
}

CSVReader::~CSVReader() {
  if (this->mFile) {
    this->mFile.close();
  }
  this->mColumnNames.clear();
}

std::vector<std::string> CSVReader::read_line() {
  std::vector<std::string> cells(this->mColumnNames.size());
  cells.clear();  // this won't change its capacity

  std::string theline;
  if (!std::getline(this->mFile, theline)) {
    return cells;
  }

  char* start = &theline.front();
  char* current = start;
  char* end = &theline.back();

  while (current < end) {
    if (*current == this->mSeparator) {
      *current = '\0';
      cells.push_back(start);
      start = current + 1;
    }
    current++;
  }
  return cells;
}

int CSVReader::num_columns() { return this->mColumnNames.size(); }

std::string CSVReader::toString(){
  std::string rep;
  for(auto s: this->mColumnNames){
    rep.append(s);
    rep.append(", ");
  }
  rep.pop_back();
  rep.pop_back();
  return rep;
}