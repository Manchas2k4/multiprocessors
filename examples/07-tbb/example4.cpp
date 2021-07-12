// =================================================================
//
// File: Example3.cpp
// Authors:
// Description: This file contains the code to count the number of
//				even numbers within an array using Intel's TBB.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include "utils.h"

#define SIZE 1000000000 //1e9

using namespace std;
using namespace tbb;

// place your code here
