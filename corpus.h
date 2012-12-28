#pragma once
#include "types.h"
#include <string>

using std::string;

class Corpus {
protected:
  index_type in_dim, out_dim;
  index_type n_examples;
public:
  bool dense;

  Corpus(const string& input_data_file, const string& output_data_file);
  Corpus();
  virtual ~Corpus() {};

  size_t size() const { return n_examples; }
  index_type inputDim() const { return in_dim; }
  index_type outputDim() const { return out_dim; }
};
