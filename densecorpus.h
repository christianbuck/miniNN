#pragma once
#include "corpus.h"

#include <stdio.h>
#include <stdlib.h>

#include <sstream>
#include <vector>
#include <string>

using std::istringstream;
using std::vector;
using std::string;

class DenseCorpus: public Corpus {
public:
  vector<DenseVector> input_data, output_data;

  DenseCorpus(const string& input_data_file, const string& output_data_file) {
    dense = true;
    loadData(input_data_file, &input_data);
    loadData(output_data_file, &output_data);

    if (input_data.size() != output_data.size()) {
      printf("Error: Input data has %lu lines, but output data has %lu lines.\n",
          input_data.size(), output_data.size());
      return;
    }

    n_examples = input_data.size();
    in_dim = input_data[0].rows();
    out_dim = output_data[0].rows();
  }

  virtual ~DenseCorpus() {};

  const DenseVector& getInputs(const size_t idx) const { return input_data.at(idx); }

  const DenseVector& getOutputs(const size_t idx) const { return output_data.at(idx); }

private:
  void loadData(const string &data_file_name, vector<DenseVector> *data) {
    FILE *data_file = fopen(data_file_name.c_str(), "r");
    if (!data_file) {
      printf("Error: Could not open %s\n", data_file_name.c_str());
      return;
    }
    char *line = NULL;
    size_t len;
    getline(&line, &len, data_file);

    bool first_line = true;
    int dim = 0;
    n_examples = 0;

    while (!feof(data_file)) {
      n_examples++;
      vector <string> parts = split(line);
      if (first_line) {
        dim = parts.size();
      }

      DenseVector v(dim);
      for (unsigned i=0; i<parts.size(); ++i) {
        v(i) = atof(parts[i].c_str());
      }
      data->push_back(v);

      getline(&line, &len, data_file);
    }

    fclose(data_file);
  }

  string strip(const string &s) {
	  unsigned long lastnonwhitespace = s.find_last_not_of("\n\t ");
	  unsigned long firstnonwhitespace = s.find_first_not_of("\n\t ");
	  unsigned long count = lastnonwhitespace - firstnonwhitespace + 1;
	  if (lastnonwhitespace != string::npos) {
		  return s.substr(firstnonwhitespace, count);
	  } else {
		  return ""; // String contains only whitespace
	  }
  }

  vector<string> split(string s) {
	  vector<string> res;
	  s = strip(s);
	  if (s.size() == 0) {
		  return res;
	  }

	  istringstream i(s);
	  while (!i.eof()) {
		  string token;
		  i >> token;
		  res.push_back(token);
	  }
	  return res;
  }

  // Fisher-Yates shuffle
  void shuffle() {
    const size_t n = size();
    for (size_t i = n - 1; i > 0; --i) {
      const size_t j = rand() % (i + 1); // 0 <= j <= i
      std::swap(input_data[i], input_data[j]);
      std::swap(output_data[i], output_data[j]);
    }
  }
};
