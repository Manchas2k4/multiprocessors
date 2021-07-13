// =================================================================
//
// File: example7.cpp
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				enumeration sort algorithm using Intel's TBB.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "utils.h"

const int SIZE = 100000;

using namespace std;
using namespace tbb;

// place your code here
