// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "mpi.h"

#include <gtest/gtest.h>
#include "ngraph/runtime/hetr/hetr.hpp"

using namespace std;

TEST(hetr, load_mpi_test)
{
    MPI::Status stat;
    MPI::Init();
    MPI::Finalize();
}

TEST(hetr, init_mpi)
{
    ngraph::runtime::Hetr hetr;
    hetr.init_mpi();
    hetr.test_macro();
    hetr.finalize_mpi();
}
