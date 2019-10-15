#ifndef __ML_UUTILS_H
#define __ML_UUTILS_H

#include <exception>

class NotImplementedError : public std::exception {
 public:
  NotImplementedError() {}
  const char* what() const noexcept override {
    return "NotImplementedError";
  }
};

class IndexOutOfBoundError : public std::exception {
 public:
  IndexOutOfBoundError() {}
  const char* what() const noexcept override {
    return "IndexOutOfBoundError";
  }
};

class FileOpenError : std::exception {
 public:
  FileOpenError() {}
  const char* what() const noexcept override {
    return "FileOpenError";
  }
};

#endif  //__ML_UUTILS_H