#include <stdio.h>
#include <stdlib.h>

#include "CSV.h"

int main(int argc, char const *argv[]){
    CSV csvframe("mobile-price-classification/train.csv", true, ',');

    csvframe.read();

    puts(csvframe.toString().c_str());

    return 0;
}
