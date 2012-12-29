#include <iostream>
#include <iomanip>
#include <string>
#include <argtable2.h>
#include <regex.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "types.h"
#include "util.h"
#include "nn.h"
#include "sparsecorpus.h"
#include "densecorpus.h"

using namespace std;

void printStats(const size_t iteration, const NN& nn, const Metrics& me, const string& prefix) {
  const int width = 7;
  cout << prefix << fixed << setprecision(4)
      << " mse: " << setw (width) << me.getMse()
      << " er: " << setw (width) << me.getErrorRate()
      << " precision: " << setw (width) << me.getPrecision()
      << " recall: " << setw (8) << me.getRecall()
      << " F1: " << setw (width) << me.getFMeasure()
      << " TN: " << setw (width) << me.getTN()
      << " FN: " << setw (width) << me.getFN()
      << " FP: " << setw (width) << me.getFP()
      << " TP: " << setw (width) << me.getTP()
      << " Delta: " << setw (width) << nn.getMagnitudeOfLastUpdate()
      << endl;
}

void adjustLearningRate(const double old_mse, const double mse, double& learningrate, const double min_learningrate=0.5) {
  if (old_mse <= mse) {
    learningrate *= 0.5;
  } else {
    learningrate *= 1.1;
  }
  learningrate = min(learningrate, 0.5);
}

void evalOnDev(NN& nn, Corpus const * const dev) {
  /*
  nn.resetDevMetrics();
  const size_t devCorpusSize = dev.size();
  for (size_t idx = 0; idx < devCorpusSize; ++idx) {
    Sentence tgt, src;
    dev.getEntry(idx, tgt, src);
    nn.evalModelDev(src, tgt);
  }
  printStats(0, nn, nn.getDevMetrics(), string("DEV: "));
  */
}

void trainNN(const size_t n_inputs, const size_t n_outputs, const size_t n_hidden, const size_t n_iters,
    const size_t corpus_size, Corpus const * const c, const size_t speed, const string& modelprefix,
    const string& loadModel, double& learningrate, const double& momentumrate, 
    Corpus const * const dev, const double regfactor=0.0, const size_t batch_size=1,
    const size_t writeEvery=0, const size_t writeEveryIteration=0, const size_t evalEvery=0)
    {
  NN nn = (loadModel.length()>0) ? NN(loadModel) : NN(n_inputs, n_outputs, n_hidden);
  nn.setLearningRate(learningrate);
  nn.setMomentumRate(momentumrate);
  nn.setRegularization(regfactor);

  double old_mse = 0.0;
  for (size_t iteration = 0; iteration < n_iters; ++iteration) {
    cout << "### Iteration " << iteration << " ###" << endl;
    nn.resetTrainMetrics();
    nn.resetGradient();
    size_t n_examples = 0;
    for (size_t idx = 0; idx < corpus_size; idx+=batch_size) {
      for (size_t batch_idx = 0; batch_idx < batch_size && idx+batch_idx < corpus_size; ++batch_idx) {
        if (c->dense) {
          const DenseVector &in_vec =
              static_cast<DenseCorpus const * const >(c)->getInputs(
                  idx + batch_idx);
          const DenseVector &out_vec =
              static_cast<DenseCorpus const * const >(c)->getOutputs(
                  idx + batch_idx);
          nn.updateGradient(in_vec.array(), out_vec.array());
        } else {
          const SparseVector &in_vec =
              static_cast<SparseCorpus const * const>(c)->getInputs(idx + batch_idx);
          const SparseVector &out_vec =
              static_cast<SparseCorpus const * const>(c)->getOutputs(idx + batch_idx);
          cout << "in " << in_vec;
          cout << "out " << out_vec;
          nn.updateGradientSparse(in_vec, out_vec);
        }

        n_examples += 1;
        progress(n_examples, corpus_size, speed);
      }
      nn.updateParameters();
      nn.resetGradient();
      if (evalEvery > 0 && n_examples % evalEvery < batch_size && dev->size() > 0) {
          evalOnDev(nn, dev);
          nn.resetDevMetrics();
      }
      if (writeEvery > 0 && n_examples % writeEvery < batch_size) {
        string modelfile = modelprefix + ".iter" + to_string(iteration) + ".idx" + to_string(n_examples) + ".model";
        cout << "writing model to " << modelfile << std::endl;
        nn.writeModelToFile(modelfile);
      }
    }

    printStats(iteration, nn, nn.getTrainMetrics(), string("TRAIN: "));
    double mse = nn.getTrainMetrics().getMse();
    if (iteration > 0) {
      adjustLearningRate(old_mse, mse, learningrate);
      nn.setLearningRate(learningrate);
    }
    std::cout << "learningrate: " << learningrate << std::endl;
    old_mse = mse;

    if (iteration == n_iters-1 || (writeEveryIteration > 0 && (iteration + 1) % writeEveryIteration == 0)) {
      string modelfile = modelprefix + ".iter" + to_string(iteration+1) + ".model";
      cout << "writing model to " << modelfile << std::endl;
      nn.writeModelToFile(modelfile);
    }
  }
}

int main(int argc, char**argv) {
  srand ( time(NULL) );
  const char *progname = "train";
  struct arg_file *Asource = arg_file1(NULL, NULL, "<sourcefile>", "input file (source corpus)");
  struct arg_file *Atarget = arg_file1(NULL, NULL, "<targetfile>", "input file (target corpus)");
  struct arg_file *Adev_source = arg_file0(NULL, "devs", "<dev-sourcefile>", "input file (dev source corpus)");
  struct arg_file *Adev_target = arg_file0(NULL, "devt", "<dev-targetfile>", "input file (dev target corpus)");
  struct arg_file *Amodel = arg_file1("s", "save", "<modelprefix>", "output file prefix (model)");
  struct arg_file *Aload = arg_file0(NULL, "load", "<modelfile>", "initial model parameters");
  struct arg_rex *Aformat = arg_rex0("f", "format", "(sparse)|(dense)", "sparse|dense", REG_ICASE|REG_EXTENDED, "Input file format. Default: sparse");
  struct arg_int *Aiters = arg_int0("i", "maxiter", "<n>", "maximum number of iterations (default 100)");
  Aiters->ival[0] = 100;
  struct arg_int *Ahidden = arg_int0("h", "hidden", "<n>", "number of hidden nodes (default 500)");
  Ahidden->ival[0] = 500;
  struct arg_int *Aeevery = arg_int0("e", "eval-every", "<n>", "evaluate on dev data every n train examples (default 10000)");
  Aeevery->ival[0] = 10000;
  struct arg_int *Awevery = arg_int0("w", "write-every", "<n>", "write model every n train examples");
  Awevery->ival[0] = 0;
  struct arg_int *Aweveryi = arg_int0(NULL, "write-every-iteration", "<n>", "write model every n iterations");
  Aweveryi->ival[0] = 0;
  struct arg_dbl *Areg = arg_dbl0("r", "regularization", "<d>", "L2 regularization parameter (default 0.0)");
  Areg->dval[0] = 0.0;
  struct arg_dbl *Alr = arg_dbl0("l", "learningrate", "<d>", "initial learning rate (default 0.5)");
  Alr->dval[0] = 0.5;
  struct arg_dbl *Am = arg_dbl0("m", "momentum", "<d>", "momentum factor (default 0.0)");
  Am->dval[0] = 0.0;
  struct arg_int *Aspeed = arg_int0(NULL, "progress", "<n>", "print dot every n examples (default 100)");
  Aspeed->ival[0] = 100;
  struct arg_int *Ab = arg_int0("b", "batch-size", "<n>", "mini-batch size (default 1)");
  Ab->ival[0] = 1;
  struct arg_lit *Averb = arg_lit0("v", "verbose", "verbose output");
  struct arg_lit *Ashuffle = arg_lit0(NULL, "shuffle", "shuffle corpus before training");
  struct arg_lit *Ahelp = arg_lit0("h", "help", "show help and exit");
  struct arg_end *Aend = arg_end(20);

  void *argtable[] = { Asource, Atarget, Adev_source, Adev_target, Amodel, Aload, Aformat, Aiters, Ahidden, Awevery, Aweveryi, Aeevery, Ab, Areg, Alr, Am, Aspeed, Averb, Ashuffle, Ahelp, Aend };
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

  const bool dense = (Aformat->count > 0) && (strcasecmp(Aformat->sval[0], "dense") == 0);

  Corpus *c;
  if (!dense) {
    c = new SparseCorpus(string(Asource->filename[0]), string(Atarget->filename[0]));
  } else {
    c = new DenseCorpus(string(Asource->filename[0]), string(Atarget->filename[0]));
  }
  if (Ashuffle->count >0) {
    c->shuffle();
  }
  const size_t corpus_size = c->size();
  size_t n_inputs = c->inputDim();
  size_t n_outputs = c->outputDim();

  const bool haveDevData = Adev_source->count >0 && Adev_target->count > 0;
  Corpus *dev;
  if (haveDevData) {
    cerr << "loading dev corpus ..." << endl;
    if (!dense) {
      dev = new SparseCorpus(string(Adev_source->filename[0]), string(Adev_target->filename[0]));
    } else {
      dev = new DenseCorpus(string(Adev_source->filename[0]), string(Adev_target->filename[0]));
    }
    n_inputs = max(c->inputDim(), dev->inputDim());
    n_outputs = max(c->outputDim(), dev->outputDim());
  } else {
    dev = new SparseCorpus();
  }

  cerr << "nin: " << n_inputs << ", nout: " << n_outputs << endl;

  const size_t n_hidden = Ahidden->ival[0];
  const size_t n_iters = Aiters->ival[0];
  const size_t writeEvery = Awevery->ival[0];
  const size_t writeEveryIteration = Aweveryi->ival[0];
  const size_t evalEvery = Aeevery->ival[0];
  double learningrate = Alr->dval[0];
  double momentumrate = Am->dval[0];
  const double regfactor = Areg->dval[0];
  const size_t speed = Aspeed->ival[0];
  const string modelfile(Amodel->filename[0]);
  const string loadModel = (Aload->count > 0) ? string(Aload->filename[0]) : string("");
  const size_t batch_size = Ab->ival[0];
  trainNN(n_inputs, n_outputs, n_hidden, n_iters, corpus_size, c, speed, modelfile, loadModel, learningrate, momentumrate, dev, regfactor, batch_size, writeEvery, writeEveryIteration, evalEvery);

  return 0;
}
