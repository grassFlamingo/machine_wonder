#include <stdio.h>
#include <stdlib.h>

#include "CSVReader.h"

#include <iostream>

int main(int argc, char const *argv[]) {
  CSVReader csvframe("mobile-price-classification/train.csv", false, ',');

  puts(csvframe.toString().c_str());

  puts("\n");
  

  return 0;
}
