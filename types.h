#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Sparse>

// typedef unsigned short index_type;
typedef unsigned int index_type;
typedef std::vector<index_type> Sentence;
typedef std::vector<Sentence> Document;
typedef float realnumber;
typedef std::vector<Sentence> Batch;

#define BATCH_SIZE 2

typedef Eigen::MatrixXf Matrix2d;
typedef Eigen::ArrayXXf Array2d;
typedef Eigen::ArrayXf Array1d;

//typedef Eigen::MatrixXd Matrix2d;
//typedef Eigen::ArrayXXd Array2d;
//typedef Eigen::ArrayXd Array1d;

typedef Eigen::SparseVector<realnumber> SparseArray1d;
typedef Eigen::SparseMatrix<realnumber> SparseMatrix2d;
