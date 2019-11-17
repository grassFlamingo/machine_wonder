/**
 *
 * We use lastnames.csv
 * - Continuous Bag-of-Words, or CBOW model.
 * - Continuous Skip-Gram Model
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toolkit.h"

int main(int argc, char const* argv[]) {
  printf("Hello character embedding!\n");

  const char* filename = "lastnames.csv";
  CNameReader cnr;
  try {
    cnr.open_file(filename);
  } catch (WEExceptions& e) {
    printf("catch an exception %s\n", WEExceptionNames[e]);
  }

  char tName[32];
  uint8 tIndex[32];
  int maxlen = -1;
  int len = 0;
  while ((len = cnr.next_name(tName)) != EOF) {
    namechar_to_index_array(tName, tIndex, len);
    printf("%s [%d", tName, tIndex[0]);
    for (int i = 1; i < len; i++)
    {
      printf(",%d", tIndex[i]);
    }
    puts("]");
    if (len > maxlen) {
      maxlen = len;
    }
  }
  printf("I found that the maximun length is %d\n", maxlen);
  return 0;
}
