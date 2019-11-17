#ifndef __H_WE_TOOLKIT_H
#define __H_WE_TOOLKIT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef u_int8_t uint8;

enum WEExceptions {
  FILE_CANNOT_OPEN = 0,
  FILE_STREAM_CLOSED,

  NAME_CHAR_NOT_SUPPORTED,
};

static const char* WEExceptionNames[] = {
    "Cannot open file",
    "File stream closed",
    "The character is not supported",
};

class CNameReader {
 public:
  CNameReader();
  CNameReader(const char* filename);
  ~CNameReader();

  void open_file(const char* filename);
  void close_file();
  int next_name(char* receiver);

 private:
  FILE* mStream;
};

uint8 namechar_to_index(char namechar);
void namechar_to_index_array(char *namechar, uint8* index, size_t len);



#endif  //__H_WE_TOOLKIT_H