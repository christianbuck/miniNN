#pragma once
#include <functional>
#include "types.h"

class ActivationFunction {
public:
  virtual Array1d sigmoid(const Array1d& inputs) const = 0;
  virtual Array1d sigmoidDeriv(const Array1d& inputs) const = 0;
  virtual realnumber getThreshold() const = 0;
  virtual realnumber maxValue() const = 0;
  virtual realnumber minValue() const = 0;
  virtual ~ActivationFunction () {};
};

class SigmoidActivationFunction: public ActivationFunction {
public:
  Array1d sigmoid(const Array1d& inputs) const {
    return inputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(standardSigmoidDouble));
  }

  Array1d sigmoidDeriv(const Array1d& inputs) const {
    return inputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(standardSigmoidDoubleDeriv));
  }

  realnumber getThreshold() const { return 0.5; }
  realnumber maxValue() const { return 1.0; }
  realnumber minValue() const { return 0.0; }

protected:
  static realnumber standardSigmoidDouble(const realnumber x) {
    if (x < -45) return 0.0;
    else if (x > 45) return 1.0;
    else return (realnumber) 1.0 / ((realnumber) 1.0 + exp(-x));
  }

  static realnumber standardSigmoidDoubleDeriv(const realnumber x) {
    return x * (1 - x);
  }

};

class TanhActivationFunction: public ActivationFunction {
public:
  Array1d sigmoid(const Array1d& inputs) const {
    return inputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(tanhSigmoidDouble));
  }

  Array1d sigmoidDeriv(const Array1d& inputs) const {
    return inputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(tanhSigmoidDoubleDeriv));
  }

  realnumber getThreshold() const { return 0.0; }
  realnumber maxValue() const { return 1.0; }
  realnumber minValue() const { return -1.0; }

protected:
  static realnumber tanhSigmoidDouble(const realnumber x) {
    return tanh(x);
  }

  static realnumber tanhSigmoidDoubleDeriv(const realnumber x) {
    return (realnumber) 1.0 - x*x;
  }
};
