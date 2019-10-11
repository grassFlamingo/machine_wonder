#include "CSV.h"

/**
 * @param filename
 * @head(bool) whether thereis a head line in CSV file
 * @separator(char) the separator of each field, usually `,`
 */
CSV::CSV(std::string filename, bool head, char separator) {
  this->mFilename = filename;
  this->mHead = head;
  this->mSeparator = separator;
}

CSV::~CSV() {}

bool CSV::read() {
  std::ifstream infile;
  infile.open(this->mFilename, std::ios::in);

  if (!infile) {
    return false;
  }
  bool result = false;

  std::string theline;

  // read head
  if (this->mHead && std::getline(infile, theline)) {
    char* hpoint = &theline.front();
    char* tpoint = hpoint;
    while (tpoint <= theline.cend().base()){
      if(*tpoint == this->mSeparator || *tpoint == '\0'){
        *tpoint = '\0';
        this->mColumnNames.push_back(hpoint);
        hpoint = tpoint+1;
      }
      tpoint++;
    }
  }

  result = true;
// csv_read_end:
  infile.close();
  return result;
}


std::string CSV::toString(){
  std::string out;
  for(std::string s: this->mColumnNames){
    out.append(s + "\n");
  }
  return out;
}