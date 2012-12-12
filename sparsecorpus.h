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

using namespace std;

class Corpus {
public:

private:
  Document src, tgt;
  index_type max_src, max_tgt;

  size_t n_words(const Document& d) const {
    size_t n = 0;
    for (Document::const_iterator it = d.begin(); it != d.end(); ++it) {
      n += it->size();
    }
    return n;
  }

  index_type readFile(const string& filename, Document& content) {
    ifstream infile(filename.c_str());
    index_type maxIdx = 0;
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
        Sentence(snt).swap(snt); // shrink to fit
        content.push_back(snt);
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
  Corpus() : max_src(0), max_tgt(0) {
  }

  Corpus(const string& src_file, const string& tgt_file) : max_src(0), max_tgt(0) {
    max_src = readFile(src_file, src);
    max_tgt = readFile(tgt_file, tgt);
    cerr << "read " << src.size() << " (src) / " << tgt.size() << " (tgt) sentences" << endl;
    cerr << "total words src: " << n_words(src) << " tgt: " << n_words(tgt) << endl;
    cerr << "max word src: " << max_src << " tgt: " << max_tgt << endl;
    assert(tgt.size() == src.size());
  }

  Corpus(const string& src_file) : max_src(0), max_tgt(0) {
    max_src = readFile(src_file, src);
    cerr << "read " << src.size() << " (src) sentences" << endl;
    cerr << "total words src: " << n_words(src) << endl;
    cerr << "max word src: " << max_src << endl;
  }

  void getEntry(const size_t idx, Sentence& s_tgt, Sentence& s_src) const {
    s_tgt = tgt.at(idx);
    s_src = src.at(idx);
  }

  void getSrcEntry(const size_t idx, Sentence& s_src) const {
    s_src = src.at(idx);
  }

  void getTgtEntry(const size_t idx, Sentence& s_tgt) const {
    s_tgt = tgt.at(idx);
  }

  size_t size() const {
    assert(tgt.empty() || tgt.size() == src.size());
    return src.size();
  }

  // Fisher-Yates shuffle
  void shuffle() {
    const size_t n = size();
    for (size_t i = n-1; i > 0; --i) {
      const size_t j = rand() % (i+1); // 0 <= j <= i
      swap(src[i], src[j]);
      swap(tgt[i], tgt[j]);
    }
  }

  index_type sourceDim() const { return max_src + 1; }
  index_type targetDim() const { return max_tgt + 1; }

};

