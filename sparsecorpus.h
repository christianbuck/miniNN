#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <limits>

#include "types.h"
#include "corpus.h"

using std::vector;
using std::ifstream;
using std::istringstream;
using std::cerr;
using std::endl;

class SparseCorpus: public Corpus {
private:
  vector<SparseVector> input_data, output_data;
  index_type max_src, max_tgt;

  size_t n_words(const vector<SparseVector>& data) const {
    size_t n = 0;
    for (vector<SparseVector>::const_iterator it = data.begin(); it != data.end(); ++it) {
      n += it->nonZeros();
    }
    return n;
  }

  index_type readFile(const string& filename, vector<SparseVector>& content) {
    ifstream infile(filename.c_str());
    index_type maxIdx = 0;
    if (infile.is_open()) {
      size_t wordIdx;
      while (infile >> wordIdx) {
        if (wordIdx > maxIdx) {
          maxIdx = wordIdx;
        }
      }
      infile.close();
    }
    infile.open(filename.c_str());
    if (infile.is_open()) {
      while (infile.good()) {
        size_t linenr = 0;
        size_t wordIdx;
        Sentence snt;

        string line;
        getline(infile, line);
        istringstream iss(line);

        while (iss >> wordIdx) {
          assert(wordIdx < std::numeric_limits<index_type>::max());
          snt.push_back(index_type(wordIdx));
          if (maxIdx < index_type(wordIdx)) {
            maxIdx = index_type(wordIdx);
          }
        }

        if (snt.empty()) {
          if (infile.good()) {
            cerr << "Warning, empty sentence in " << filename
                << " line:" << linenr << " (not skipping)" << endl;
          } else {
            continue;
          }
        }

        // transform into bag-of-words
        std::sort(snt.begin(), snt.end());
        Sentence::iterator it;
        it = unique(snt.begin(), snt.end());
        snt.resize(it - snt.begin());
        SparseVector sv(maxIdx+1);
        sv.reserve(snt.size());
        for (Sentence::const_iterator it = snt.begin(); it != snt.end(); ++it) {
          sv.insertBack(*it) = 1.0;
        }
        cerr << sv.rows() << " " << sv.cols() << endl;
        content.push_back(sv);
        ++linenr;
      }
      infile.close();
    } else {
      cerr << "Unable to open file" << filename << endl;
    }
    cerr << "maxIdx: " << maxIdx << endl;
    return maxIdx;
  }

public:
//  SparseCorpus() : in_dim(0), out_dim(0), n_examples(0) {
//  }

  SparseCorpus(const string& input_data_file, const string& output_data_file) {
    dense = false;
    in_dim = readFile(input_data_file, input_data) + 1;
    out_dim = readFile(output_data_file, output_data) + 1;
    cerr << "read " << input_data.size() << " (in) / " << output_data.size() << " (out) examples" << endl;
    cerr << "total words src: " << n_words(input_data) << " tgt: " << n_words(output_data) << endl;
    cerr << "dimensions: " << in_dim << " out: " << out_dim << endl;
    assert(output_data.size() == input_data.size());
    n_examples = output_data.size();
  }

  SparseCorpus(const string& src_file) {
    dense = false;
    in_dim = readFile(src_file, input_data);
    cerr << "read " << input_data.size() << " (src) sentences" << endl;
    cerr << "total words src: " << n_words(input_data) << endl;
    cerr << "max word src: " << in_dim << endl;
  }

  SparseCorpus() {}

  void getEntry(const size_t idx, SparseVector& in_vec, SparseVector& out_vec) const {
    in_vec = output_data.at(idx);
    out_vec = input_data.at(idx);
  }

  const SparseVector& getInputs(const size_t idx) const { return input_data.at(idx); }

  const SparseVector& getOutputs(const size_t idx) const { return output_data.at(idx); }

  // Fisher-Yates shuffle
  void shuffle() {
    const size_t n = size();
    for (size_t i = n-1; i > 0; --i) {
      const size_t j = rand() % (i+1); // 0 <= j <= i
      std::swap(input_data[i], input_data[j]);
      std::swap(output_data[i], output_data[j]);
    }
  }

};

