#ifndef __ML_UTILS_H
#define __ML_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>

#include "utils.h"
#include "theColumn.h"

class CSVReader {
 private:
  std::vector<std::string> mColumnNames;
  char mSeparator;
  std::ifstream mFile;

 public:
  /**
   * @param filename
   * @usehead(bool) whether thereis a head line in CSV file
   * @separator(char) the separator of each field, usually `,`
   */
  CSVReader(const char* filename, bool usehead, char separator);
  ~CSVReader();

  std::vector<std::string> read_line();

  int num_columns();
  std::string toString();
};

#endif  //__ML_UTILS_H