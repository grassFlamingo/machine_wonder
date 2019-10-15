#include <stdio.h>
#include <stdlib.h>

#include "CSVReader.h"
#include "ColumnTable.h"

int main(int argc, char const *argv[]) {
  CSVReader csvframe("mobile-price-classification/train.csv", true, ',');

  puts(csvframe.toString().c_str());

  std::vector<NoneColumn*> theColumns(csvframe.num_columns());
  for(size_t i = 0; i < theColumns.size(); i++){
    // sorry we use Float32 for all
    theColumns[i] = new Float32Column(csvframe.column_name(i));
  }

  while(true){
    auto line = csvframe.read_line();
    if(line.empty()){
      break;
    }
    for(size_t i = 0; i < line.size(); i++){
      theColumns[i]->append(line[i]);
    }
  }

  for(size_t i = 0; i < 10; i++){
    for(size_t j = 0; j < theColumns.size(); j++){
      printf("%.2f ", theColumns[j]->get_float32(i));
    }
    printf("\n");
  }

  for(auto j: theColumns){
    printf("mean of %s is %f\n", j->name_cstr(), j->mean());
  }

  for(auto c: theColumns){
    delete c;
  }
  theColumns.clear();
  return 0;
}
