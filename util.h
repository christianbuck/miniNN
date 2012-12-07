#ifndef UTIL_H_
#define UTIL_H_

#include <sstream>
#include <iostream>
#include <vector>

void usage(void **argtable, const char *progname) {
  FILE *fp = stdout;
  fprintf(fp, "Usage: %s ", progname);
  arg_print_syntaxv(fp, argtable, "\n");
  arg_print_glossary(fp, argtable, " %-50s %s\n");
}

template<class T>
std::string vectorToString(const std::vector<T>& v) {
  std::stringstream ss;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) {
      ss << " ";
    }
    ss << v[i];
  }
  return ss.str();
}

template<class T>
inline std::string to_string(const T& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

void progress(const size_t iter, const size_t corpus_size, const size_t speed, const bool force = false) {
  const size_t true_iter = iter % corpus_size;
  if (true_iter > 0 && true_iter % speed == 0) {
    std::cerr << ".";
    std::cerr.flush();
  }

  if ((true_iter > 0 && true_iter % (50 * speed) == 0) != (force)) {
    std::cerr << "[" << iter % corpus_size << "]" << std::endl;
    std::cerr.flush();
  }
}

template<class T>
void printVector(const std::vector<T>& v, const double threshold=0.5, const bool verbose=false) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) {
      std::cout << " ";
    }
    std::cout << (v.at(i) > threshold);
    if (verbose)
      std::cout << " (" << v.at(i) << ")";
  }
  std::cout << std::endl;
}


#endif /* UTIL_H_ */
