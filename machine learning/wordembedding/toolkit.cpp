#include "toolkit.h"

CNameReader::CNameReader() { this->mStream = NULL; }

CNameReader::CNameReader(const char* filename) { this->open_file(filename); }

CNameReader::~CNameReader() { this->close_file(); }

void CNameReader::open_file(const char* filename) {
  this->close_file();
  this->mStream = fopen(filename, "r");
  if (this->mStream == NULL) {
    throw WEExceptions::FILE_CANNOT_OPEN;
  }
}

void CNameReader::close_file() {
  if (this->mStream != NULL) {
    fclose(this->mStream);
  }
}

int CNameReader::next_name(char* receiver) {
  if (this->mStream == NULL) {
    throw WEExceptions::FILE_STREAM_CLOSED;
  }
  char tc;
  int i = 0;
  while (true) {
    tc = fgetc(this->mStream);
    if (tc == EOF) {
      return EOF;
    } else if (tc == ',') {
      receiver[i] = '\0';
      return i;
    } else {
      receiver[i++] = tc;
    }
  }
  return i;
}

// functions

uint8 namechar_to_index(char namechar) {
  if (namechar < 'A') {
    throw WEExceptions::NAME_CHAR_NOT_SUPPORTED;
  } else if (namechar <= 'Z') {
    return namechar - 'A';
  } else if (namechar < 'a') {
    throw WEExceptions::NAME_CHAR_NOT_SUPPORTED;
  } else if (namechar <= 'z') {
    return namechar - 'a';
  } else {
    throw WEExceptions::NAME_CHAR_NOT_SUPPORTED;
  }
}

void namechar_to_index_array(char* namechar, uint8* index, size_t len) {
  for (size_t i = 0; i < len; i++) {
    index[i] = namechar_to_index(namechar[i]);
  }
}

