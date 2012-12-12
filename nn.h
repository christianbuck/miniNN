#pragma once
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Sparse>
#include "types.h"
using std::vector;

typedef Eigen::MatrixXf Matrix2d;
typedef Eigen::ArrayXXf Array2d;
typedef Eigen::ArrayXf Array1d;
//
//typedef Eigen::MatrixXd Matrix2d;
//typedef Eigen::ArrayXXd Array2d;
//typedef Eigen::ArrayXd Array1d;

typedef Eigen::SparseVector<realnumber> SparseArray1d;
typedef Eigen::SparseMatrix<realnumber> SparseMatrix2d;

// regular sigmoid is now tanh
realnumber sigmoidDouble(const realnumber x) {
 /* if (x < -45) return 0.0;
  else if (x > 45) return 1.0;
  else return (realnumber) 1.0 / ((realnumber) 1.0 + exp(-x));
 */
  return tanh(x);
}

realnumber sigmoidDoubleDeriv(const realnumber x) {
  // return x * (1 - x);
  return (realnumber) 1.0 - x*x;
}

class Gradient {
public:
  Array2d gt1, gt2;
  Array1d gb1, gb2;
  Gradient() {
  }

  Gradient(const Gradient& g) {
    std::cerr << "Coying gradient" << std::endl;
    gt1 = g.gt1;
    gt2 = g.gt2;
    gb1 = g.gb1;
    gb2 = g.gb2;
  }
  
  Gradient& operator=(const Gradient& g) {
    gt1 = g.gt1;
    gt2 = g.gt2;
    gb1 = g.gb1;
    gb2 = g.gb2;
    return *this;
  }

  realnumber abs() const {
    return gt1.abs().sum() + gt2.abs().sum() + gb1.abs().sum() + gb2.abs().sum();
  }

  Gradient(const size_t n_in, const size_t n_out, const size_t n_hidden) {
    std::cerr << "Creating fixed size gradient" << std::endl;
    reset(n_in, n_out, n_hidden);
  }

  void set_size(const size_t n_in, const size_t n_out, const size_t n_hidden) {
    gt1 = Array2d::Zero(n_hidden, n_in);
    gt2 = Array2d::Zero(n_out, n_hidden);
    gb1 = Array1d::Zero(n_hidden);
    gb2 = Array1d::Zero(n_out);
  }

  void reset(const size_t n_in, const size_t n_out, const size_t n_hidden) {
    gt1 = Array2d::Zero(n_hidden, n_in);
    gt2 = Array2d::Zero(n_out, n_hidden);
    gb1 = Array1d::Zero(n_hidden);
    gb2 = Array1d::Zero(n_out);
  }
};

class Metrics {
private:
  size_t n_examples;
  double cost;
  double squared_error;
  size_t n_errors;
  size_t tp;
  size_t fp;
  size_t tn;
  size_t fn;

public:
  Metrics() {
    reset();
  }

  void reset() {
    n_examples = 0;
    cost = 0.0;
    squared_error = 0.0;
    n_errors = 0;
    fp = 0;
    fn = 0;
    tp = 0;
    tn = 0;
  }

  void update(const Array1d& outputs, const Array1d& predictions) {
    const double l_cost = (-outputs * predictions.log() - (1 - outputs) * (1 - predictions).log()).sum();
    const double l_mse = (outputs - predictions).pow(2).sum();
    const double l_errors = ((outputs - predictions).abs() > 0.5).cast<int>().sum();
    {
      cost += l_cost;
      squared_error += l_mse;
      n_errors += l_errors;
      n_examples++;
      for (int i = 0; i < outputs.rows(); ++i) {
        if (predictions(i) <= 0.5) {
          if (outputs(i) <= 0.5)
            ++tn;
          else
            ++fn;
        } else {
          if (outputs(i) > 0.5)
            ++tp;
          else
            ++fp;
        }
      }
    }
  }

  double getCost() const {
    return cost / n_examples; // + ;
  }

  double getMse() const {
    return squared_error / n_examples;
  }

  double getErrorRate() const {
    return double(fp + fn) / double(tp + fp + tn + fn);
  }

  double getPrecision() const {
    return double(tp) / double(tp + fp);
  }

  double getRecall() const {
    return double(tp) / double(tp + fn);
  }

  double getSpecificity() const {
    return double(tn) / double(tn + fp);
  }

  size_t getTP() const { return tp; }
  size_t getTN() const { return tn; }
  size_t getFP() const { return fp; }
  size_t getFN() const { return fn; }

  double getFMeasure(const double beta=1.0) const {
    const double precision = getPrecision();
    const double recall = getRecall();
    const double bsq = beta*beta;
    return (1 + bsq) * (precision * recall) / (bsq*precision + recall);
  }

};

class NN {
private:
  size_t n_in;
  size_t n_out;
  size_t n_hidden;
  Array2d t1, t2;
  Array1d b1, b2;
  Gradient g;
  Gradient momentum;

  Metrics train_metrics, dev_metrics, eval_metrics;

  double learningrate;
  double momentumrate;
  double reg_factor;
  size_t n_examples;

  Array1d sigmoid(Array1d& x) const {
    return x.unaryExpr(std::ptr_fun(sigmoidDouble));
  }

  void checkFormat(std::istream& iss) const {
    // call this at those points where we expect a 
    // # in the data. Here we read something and
    // check if it's a #
    std::string s;
    iss >> s;
    if (s != "#") {
      std::cerr << "wrong format: " << s << std::endl;
    }
  }

  void readWeights(std::istream& iss) {
    realnumber f;

    iss >> n_in;
    iss >> n_hidden;
    iss >> n_out;

    checkFormat(iss);
    b1.resize(n_hidden);
    for (size_t i = 0; i < n_hidden; ++i) {
      iss >> f;
      b1(i) = f;
    }

    checkFormat(iss);
    t1.resize(n_hidden, n_in);
    for (size_t i = 0; i < n_hidden * n_in; ++i) {
      iss >> f;
      t1(i / n_in, i % n_in) = f;
    }

    checkFormat(iss);
    b2.resize(n_out);
    for (size_t i = 0; i < n_out; ++i) {
      iss >> f;
      b2(i) = f;
    }

    checkFormat(iss);
    t2.resize(n_out, n_hidden);
    for (size_t i = 0; i < n_out * n_hidden; ++i) {
      iss >> f;
      t2(i / n_hidden, i % n_hidden) = f;
    }
    checkFormat(iss);
  }

public:
  NN(size_t n_in, size_t n_out, size_t n_hidden) :
      n_in(n_in), n_out(n_out), n_hidden(n_hidden), 
      learningrate(0.0), momentumrate(0.0), reg_factor(0.0),
      n_examples(0) {
    // init weights to random values
    const realnumber e_init = sqrt(6) / sqrt(n_in + n_out);
    t1 = Array2d::Random(n_hidden, n_in) * e_init;
    t2 = Array2d::Random(n_out, n_hidden) * e_init;
    b1 = Array1d::Random(n_hidden) * e_init;
    b2 = Array1d::Random(n_out) * e_init;
    g = Gradient(n_in, n_out, n_hidden);
    momentum = Gradient(n_in, n_out, n_hidden);
    n_examples = 0;
  }

  NN(const std::string& filename) :
      learningrate(0.0), momentumrate(0.0), reg_factor(0.0),
      n_examples(0) {
    readModelFromFile(filename);
    momentum = Gradient(n_in, n_out, n_hidden);
    n_examples = 0;
  }

  size_t getInputDimension() const {
    return n_in;
  }

  size_t getOutputDimension() const {
    return n_out;
  }

  size_t getNHiddenNodes() const {
    return n_hidden;
  }

  void setLearningRate(const double rate) {
    learningrate = rate;
  }

  void setMomentumRate(const double rate) {
    momentumrate = rate;
  }

  double getLearningRate() const {
    return learningrate;
  }

  void setRegularization(const double lambda) {
    reg_factor = lambda;
  }

  double getRegularization() const {
    return reg_factor;
  }

  void resetGradient() {
    g.reset(n_in, n_out, n_hidden);
  }

  void forward(const Array1d& inputs, Array1d& output_activations) const {
    const Array1d hidden_activations = ((t1.matrix() * inputs.matrix()).array() + b1).unaryExpr(std::ptr_fun(sigmoidDouble));
    output_activations = ((t2.matrix() * hidden_activations.matrix()).array() + b2).unaryExpr(std::ptr_fun(sigmoidDouble));
  }

  void forward(const SparseArray1d& sparseInputs, Array1d& output_activations) const {
    const Array1d hidden_activations = ((t1.matrix() * sparseInputs).array() + b1).unaryExpr(std::ptr_fun(sigmoidDouble));
    output_activations = ((t2.matrix() * hidden_activations.matrix()).array() + b2).unaryExpr(std::ptr_fun(sigmoidDouble));
  }

  void printWeights(std::ostream& oss) const {
    oss << n_in << " " << n_hidden << " " << n_out << std::endl;
    oss << "# " << std::endl << b1.transpose() << std::endl;
    oss << "# " << std::endl << t1 << std::endl;
    oss << "# " << std::endl << b2.transpose() << std::endl;
    oss << "# " << std::endl << t2 << std::endl;
    oss << "# " << std::endl;
  }

  void printGradient() const {
    std::cout << "Grad input-to-hidden: " << std::endl;
    std::cout << g.gb1 << std::endl;
    std::cout << g.gt1 << std::endl;
    std::cout << "Grad hidden-to-output: " << std::endl;
    std::cout << g.gb2 << std::endl;
    std::cout << g.gt2 << std::endl << std::endl;
  }

  void updateGradient(const Array1d& inputs, const Array1d& outputs) {
    const Array1d hidden_activations = ((t1.matrix() * inputs.matrix()).array() + b1).unaryExpr(std::ptr_fun(sigmoidDouble));
    const Array1d predictons = ((t2.matrix() * hidden_activations.matrix()).array() + b2).unaryExpr(std::ptr_fun(sigmoidDouble));

    train_metrics.update(outputs, predictons);

    const Array1d d3 = -(outputs - predictons) * predictons.unaryExpr(std::ptr_fun(sigmoidDoubleDeriv));
    Array1d d2 = t2.matrix().transpose() * d3.matrix();
    d2 *= hidden_activations.unaryExpr(std::ptr_fun(sigmoidDoubleDeriv));

    g.gt2 += (d3.matrix() * hidden_activations.matrix().transpose()).array();
    g.gt1 += (d2.matrix() * inputs.matrix().transpose()).array();
    g.gb2 += d3;
    g.gb1 += d2;
    ++n_examples;
  }

  void updateGradient(const SparseArray1d& sparseInputs, const Array1d& outputs) {
    const Array1d hidden_activations = ((t1.matrix() * sparseInputs).array() + b1).unaryExpr(std::ptr_fun(sigmoidDouble));
    const Array1d output_activations = ((t2.matrix() * hidden_activations.matrix()).array() + b2).unaryExpr(std::ptr_fun(sigmoidDouble));

    train_metrics.update(outputs, output_activations);

    Array1d delta_output = -(outputs - output_activations);
    delta_output *= output_activations.unaryExpr(std::ptr_fun(sigmoidDoubleDeriv));
    Array1d delta_hidden = t2.matrix().transpose() * delta_output.matrix();
    delta_hidden *= hidden_activations.unaryExpr(std::ptr_fun(sigmoidDoubleDeriv));

    g.gb2 += delta_output;
    g.gt2 += (delta_output.matrix() * hidden_activations.matrix().transpose()).array();
    g.gb1 += delta_hidden;
    for (SparseArray1d::InnerIterator it(sparseInputs); it; ++it) {
      g.gt1.col(it.index()) += delta_hidden; // implicit * 1
    }
    n_examples++;
  }

  void sparseToDense(const Sentence& sparseVector, const size_t maxIdx, Array1d& denseVector) const {
    for (Sentence::const_iterator it = sparseVector.begin(); it != sparseVector.end(); ++it) {
      if ((*it) >= maxIdx) {
        std::cerr << *it << ">" << maxIdx - 1 << std::endl;
      }
      denseVector(*it) = 1.0;
    }
  }

  void sparseToDense(const Batch& sparseBatch, const size_t maxIdx, Array2d& denseData) const {
    for (size_t bidx = 0; bidx < sparseBatch.size(); ++bidx) {
      for (size_t sidx = 0; sidx < sparseBatch[bidx].size(); ++sidx) {
        if (sparseBatch[bidx][sidx] >= maxIdx) {
          std::cerr << sparseBatch[bidx][sidx] << ">" << maxIdx - 1 << std::endl;
        }
        denseData(bidx, sparseBatch[bidx][sidx]) = 1.0;
      }
    }
  }

  void sparseToSparse(const Sentence& sparseVector, SparseArray1d& sparseOutputVector) const {
    sparseOutputVector.reserve(sparseVector.size());
    for (Sentence::const_iterator it = sparseVector.begin(); it != sparseVector.end(); ++it) {
      sparseOutputVector.insertBack(*it) = 1.0;
    }
  }

  void updateGradientSparse(const Sentence& inputs, const Sentence& outputs) {
    Array1d v_out = Array1d::Zero(n_out);
    sparseToDense(outputs, n_out, v_out);
    SparseArray1d v_in_sparse(n_in);
    sparseToSparse(inputs, v_in_sparse);
    updateGradient(v_in_sparse, v_out);
  }

  void updateGradientDense(const Sentence& inputs, const Sentence& outputs) {
    Array1d v_out = Array1d::Zero(n_out);
    sparseToDense(outputs, n_out, v_out);
    Array1d v_in = Array1d::Zero(n_in);
    sparseToDense(inputs, n_in, v_in);
    updateGradient(v_in, v_out);
  }

  void evalModelDev(const Sentence& inputs, const Sentence& outputs) {
    Array1d v_out = Array1d::Zero(n_out);
    sparseToDense(outputs, n_out, v_out);

    SparseArray1d v_in_sparse(n_in);
    sparseToSparse(inputs, v_in_sparse);
    
    Array1d predictions;
    forward(v_in_sparse, predictions);
    dev_metrics.update(v_out, predictions);
  }

  void evalModel(const Sentence& inputs, vector<realnumber>& outputs) const {
    SparseArray1d v_in_sparse(n_in);
    sparseToSparse(inputs, v_in_sparse);
    const Array1d a2 = ((t1.matrix() * v_in_sparse).array() + b1).unaryExpr(std::ptr_fun(sigmoidDouble));
    const Array1d a3 = ((t2.matrix() * a2.matrix()).array() + b2).unaryExpr(std::ptr_fun(sigmoidDouble));
    outputs.resize(n_out);
    for (size_t i = 0; i < n_out; ++i) {
      outputs[i] = realnumber(a3(i));
    }
  }

  void evalModel(const Sentence& inputs, vector<vector<realnumber> >& outputs) const {
    outputs.clear();
    outputs.resize(inputs.size());

    for (size_t f1o = 0; f1o<inputs.size(); f1o++) {
      Sentence l1oinputs = inputs;
      l1oinputs.erase(l1oinputs.begin()+f1o);
      SparseArray1d v_in_sparse(n_in);
      sparseToSparse(l1oinputs , v_in_sparse);
      const Array1d a2 = ((t1.matrix() * v_in_sparse).array() + b1).unaryExpr(std::ptr_fun(sigmoidDouble));
      const Array1d a3 = ((t2.matrix() * a2.matrix()).array() + b2).unaryExpr(std::ptr_fun(sigmoidDouble));
      outputs[f1o].resize(n_out);
      for (size_t i = 0; i < n_out; ++i) {
        outputs[f1o][i] = realnumber(a3(i));
      }
    }
  }

  void evalModel2(const Sentence& inputs, vector<realnumber>& outputs) const {
    Array1d v_in = Array1d::Zero(n_in);
    sparseToDense(inputs, n_in, v_in);
    Array1d hidden_activations = ((t1.matrix() * v_in.matrix()).array() + b1).unaryExpr(std::ptr_fun(sigmoidDouble));
    Array1d prediction = ((t2.matrix() * hidden_activations.matrix()).array() + b2).unaryExpr(std::ptr_fun(sigmoidDouble));
    outputs.clear();
    outputs.resize(n_out);
    for (size_t i = 0; i < n_out; ++i) {
      outputs[i] = realnumber(prediction(i));
    }
  }

  void updateParameters() {
//    double factor = learningrate / n_examples;
//    std::cout << "nex: " << n_examples << " factor: " << factor << std::endl;
//    std::cerr << "updating weights" << std::endl;
    const double mfactor = momentumrate;
    const double factor = learningrate / n_examples;

    if (reg_factor > 0.0) {
      momentum.gt1 = g.gt1 * factor + momentum.gt1 * mfactor + t1 * (reg_factor * learningrate);
    } else {
      momentum.gt1 = g.gt1 * factor + momentum.gt1 * mfactor;
    }
    t1 -= momentum.gt1;

    if (reg_factor > 0.0) {
      momentum.gt2 = g.gt2 * factor + momentum.gt2 * mfactor + t2 * (reg_factor * learningrate);
    } else {
      momentum.gt2 = g.gt2 * factor + momentum.gt2 * mfactor;
    }
    t2 -= momentum.gt2;

    // no regularization for biases
    momentum.gb1 = g.gb1 * factor + momentum.gb1 * mfactor;
    momentum.gb2 = g.gb2 * factor + momentum.gb2 * mfactor;
    b1 -= momentum.gb1;
    b2 -= momentum.gb2;
    n_examples = 0;
  }

  void writeModelToFile(const std::string& filename) const {
    std::ofstream outfile;
    outfile.open(filename.c_str());
    printWeights(outfile);
    outfile.close();
  }

  void readModelFromFile(const std::string& filename) {
    std::ifstream infile;
    infile.open(filename.c_str());
    readWeights(infile);
    infile.close();
  }

  const Metrics& getTrainMetrics() const {
    return train_metrics;
  }

  const Metrics& getDevMetrics() const {
    return dev_metrics;
  }

  void resetDevMetrics() {
    dev_metrics.reset();
  }

  void resetTrainMetrics() {
    train_metrics.reset();
  }

  double getNorm1() const {
    return t1.matrix().lpNorm<1>() + t2.matrix().lpNorm<1>();
  }

  double getNorm2() const {
    return t1.matrix().lpNorm<2>() + t2.matrix().lpNorm<2>();
  }

  size_t getNExamples() const {
    return n_examples;
  }

  double getRegCost() const {
    return (t1.pow(2).sum() + t2.pow(2).sum()) * reg_factor;
  }

  double getMagnitudeOfLastUpdate() const {
    return momentum.abs();
  }
};
