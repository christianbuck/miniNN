#pragma once
#include <functional>
#include "types.h"

class ActivationFunction {
public:
  virtual Array1d sigmoid(const Array1d& inputs) const = 0;
  virtual Array1d sigmoidDeriv(const Array1d& inputs, const Array1d& outputs) const = 0;
  virtual realnumber getThreshold() const = 0;
  virtual realnumber maxValue() const = 0;
  virtual realnumber minValue() const = 0;
  virtual ~ActivationFunction () {};
};

class LogisticActivationFunction: public ActivationFunction {
public:
  Array1d sigmoid(const Array1d& inputs) const {
    return inputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(logisticSigmoidDouble));
  }

  Array1d sigmoidDeriv(const Array1d& inputs, const Array1d& outputs) const {
    return outputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(logisticSigmoidDoubleDeriv));
  }

  realnumber getThreshold() const { return 0.5; }
  realnumber maxValue() const { return 1.0; }
  realnumber minValue() const { return 0.0; }

protected:
  static realnumber logisticSigmoidDouble(const realnumber x) {
    if (x < -45) return 0.0;
    else if (x > 45) return 1.0;
    else return (realnumber) 1.0 / ((realnumber) 1.0 + exp(-x));
  }

  static realnumber logisticSigmoidDoubleDeriv(const realnumber x) {
    return x * (1 - x);
  }

};

class TanhActivationFunction: public ActivationFunction {
public:
  Array1d sigmoid(const Array1d& inputs) const {
    return inputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(tanhSigmoidDouble));
  }

  Array1d sigmoidDeriv(const Array1d& inputs, const Array1d& outputs) const {
    return outputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(tanhSigmoidDoubleDeriv));
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

class RectifiedActivationFunction: public ActivationFunction {
public:
  Array1d sigmoid(const Array1d& inputs) const {
    return inputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(rectSigmoidDouble));
  }

  Array1d sigmoidDeriv(const Array1d& inputs, const Array1d& outputs) const {
    return inputs.unaryExpr(std::ptr_fun<realnumber, realnumber>(rectSigmoidDoubleDeriv));
  }

  realnumber getThreshold() const { return 0.5 * (min + max); }
  realnumber maxValue() const { return max; }
  realnumber minValue() const { return min; }

private:
  static const realnumber min = 0;
  static const realnumber max = 10;

protected:
  static realnumber rectSigmoidDouble(const realnumber x) {
    if (x < -45) return 0.0;
    else if (x > 45) return x;
    return log(1 + exp(x));
    //return (realnumber) (x < 0) ? min : x;
  }

  static realnumber rectSigmoidDoubleDeriv(const realnumber x) {
    if (x < -45) return 0.0;
    else if (x > 45) return 1.0;
    else return (realnumber) 1.0 / ((realnumber) 1.0 + exp(-x));
    //return (x < 0) ? 0 : 1;
  }
};
