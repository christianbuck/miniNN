#pragma once
#include "types.h"
#include <string>

using std::string;

class Corpus {
public:
  bool dense;
  index_type in_dim, out_dim;
  index_type n_examples;

  virtual Corpus(const string& input_data_file, const string& output_data_file) = 0;
  virtual ~Corpus() {};

  size_t size() const { return n_examples; }
  index_type sourceDim() const { return in_dim; }
  index_type targetDim() const { return out_dim; }

};
