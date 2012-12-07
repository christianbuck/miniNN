#pragma once
#include <vector>

// typedef unsigned short index_type;
typedef unsigned int index_type;
typedef std::vector<index_type> Sentence;
typedef std::vector<Sentence> Document;
typedef float realnumber;
typedef std::vector<Sentence> Batch;

#define BATCH_SIZE 2
