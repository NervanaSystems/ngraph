//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <chrono>
#include <iostream>

#ifdef NGRAPH_MLIR_ENABLE
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/IR/MLIRContext.h>
#include "contrib/mlir/core/ngraph_dialect/dialect.hpp"
#include "contrib/mlir/utils.hpp"
#endif
#include "gtest/gtest.h"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"

#ifdef NGRAPH_UNIT_TEST_NUMPY_ENABLE
#include <pybind11/embed.h>
#endif

using namespace std;

#ifdef NGRAPH_UNIT_TEST_NUMPY_ENABLE
namespace py = pybind11;
#endif

int main(int argc, char** argv)
{
    const string cpath_flag{"--cpath"};
    string cpath;
    const char* exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    argv_vector.push_back(const_cast<char*>(exclude));
    for (int i = 1; i < argc; i++)
    {
        argv_vector.push_back(argv[i]);
    }
    argc = argv_vector.size();
    ::testing::InitGoogleTest(&argc, argv_vector.data());
    for (int i = 1; i < argc; i++)
    {
        if (cpath_flag == argv[i] && (++i) < argc)
        {
            cpath = argv[i];
        }
    }
    ngraph::runtime::Backend::set_backend_shared_library_search_directory(cpath);
#ifdef NGRAPH_MLIR_ENABLE
    // Initialize MLIR
    ngraph::runtime::ngmlir::initializeNGraphMLIR();
    mlir::DialectRegistry& registry = mlir::getGlobalDialectRegistry();
    registry.insert<
        // In-tree Dialects.
        mlir::AffineDialect,
        mlir::LLVM::LLVMDialect,
        mlir::scf::SCFDialect,
        mlir::StandardOpsDialect,
        mlir::vector::VectorDialect,
        // nGraph dialects.
        mlir::NGraphOpsDialect>();
#endif

#ifdef NGRAPH_UNIT_TEST_NUMPY_ENABLE
    // Setup embedded python interpreter and import numpy
    py::scoped_interpreter guard{};
    py::exec(R"(
import numpy as np
)",
             py::globals(),
             py::dict());
#endif

    int rc = RUN_ALL_TESTS();

    return rc;
}
