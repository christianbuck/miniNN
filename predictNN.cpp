#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cassert>

#include <argtable2.h>

#include "types.h"
#include "nn.h"
#include "sparsecorpus.h"
#include "util.h"

using std::vector;
using std::string;

realnumber rmse(const Sentence& observation, const vector<realnumber>& prediction) {
  vector<double> squares;
  const size_t dim = prediction.size();
  const size_t nobs = observation.size();
  squares.reserve(2*nobs + dim);

  // we assume that observation is a sparse vector and first
  // assume that all entries are zero and later fix that.
  for (size_t i = 0; i < dim; ++i) {
    squares.push_back(prediction[i] * prediction[i]);
  }

  for (size_t k = 0; k < nobs; ++k) {
    const index_type i = observation[k];
    squares.push_back(-(prediction[i] * prediction[i])); // fix error
    squares.push_back((1-prediction[i]) * (1-prediction[i]));
  }

  std::sort(squares.begin(), squares.end());

  realnumber squared_error = 0;
  for (size_t i = 0; i < squares.size(); ++i) {
    squared_error += squares[i];
  }

  return sqrt(squared_error/prediction.size());
}

int main(int argc, char**argv) {
  const char *progname = "train";
  struct arg_file *Asource = arg_file1(NULL, NULL, "<sourcefile>", "input file (source corpus)");
  struct arg_file *Aload = arg_file1(NULL, NULL, "<modelfile>", "model parameters");
  struct arg_file *Atarget = arg_file0("t", "tgt", "<targetfile>", "input file (target corpus)");
  struct arg_lit *Averb = arg_lit0("v", "verbose", "verbose output");
  struct arg_lit *Ahelp = arg_lit0("h", "help", "show help and exit");
  struct arg_end *Aend = arg_end(20);

  void *argtable[] = { Asource, Aload, Atarget, Averb, Ahelp, Aend };
  int nerrors = arg_parse(argc, argv, argtable);
  // special flags
  if (Ahelp->count > 0) {
    usage(argtable, progname);
    exit(0);
  }
  if (nerrors > 0) {
    arg_print_errors(stderr, Aend, progname);
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
    exit(EXIT_FAILURE);
  }

  NN nn(string(Aload->filename[0]));
  std::cerr << "loaded NN-model with "
      << nn.getInputDimension() << " input, "
      << nn.getNHiddenNodes() << " hidden and "
      << nn.getOutputDimension() << " output nodes" << std::endl;

  if (Atarget->count >0) {
    Corpus c(string(Asource->filename[0]), string(Atarget->filename[0]));
    realnumber tot_err = 0;
    for (size_t i = 0; i < c.size(); ++i) {
      Sentence src;
      c.getSrcEntry(i, src);
      vector<realnumber> prediction;
      nn.evalModel(src, prediction);

      Sentence tgt;
      c.getTgtEntry(i, tgt);
      realnumber err = rmse(tgt, prediction);
      tot_err += err;
      std::cout << err << std::endl;
      //printVector(prediction, .5, true);
    }
    std::cerr << "total error: " << tot_err
              << " average: "    << tot_err/c.size() <<  std::endl;
  } else {
    Corpus c(string(Asource->filename[0]));
    for (size_t i = 0; i < c.size(); ++i) {
      Sentence src;
      c.getSrcEntry(i, src);
      vector<realnumber> prediction;
      nn.evalModel(src, prediction);
      printVector(prediction, .5, true);
    }
  }

  return 0;
}
