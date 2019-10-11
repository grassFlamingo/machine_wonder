#ifndef __ML_UTILS_H
#define __ML_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <fstream>

#include "theColumn.h"

class CSV {
 private:
  BaseColumn* mColumns;
  std::vector<std::string> mColumnNames;
  std::string mFilename;
  bool mHead;
  char mSeparator;



 public:
  /**
   * @param filename
   * @head(bool) whether thereis a head line in CSV file
   * @separator(char) the separator of each field, usually `,`
   */
  CSV(std::string filename, bool head, char separator);
  ~CSV();

  bool read();

  std::string toString();
};

#endif  //__ML_UTILS_H