//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
//
/// \file `ngraph-opt` is a driver for MLIR back-end in nGraph, similar to `opt` and `mlir-opt` for
/// LLVM and MLIR, respectively. It allows invoking a sequence of arbitrary MLIR passes on a given
/// input IR. For example, `ngraph-opt my_test.mlir -optA -optC` will run `optA` and `optC`, in that
/// particular order, on the input IR in file `my_test.mlir` and dump the resulting IR to the
/// standard output.
///
/// `ngraph-opt` is used in LLVM-style LIT tests since it allows invoking a single MLIR pass or a
/// small sequence of passes without running the whole compiler pipeline. Please, refer to
/// ngraph_repo_path/tests/mlir/ for examples.

#include "contrib/mlir/compiler/tools.hpp"
#include "ngraph/check.hpp"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/MlirOptMain.h>
#include "llvm/Support/InitLLVM.h"

static llvm::cl::opt<std::string>
    input_filename(llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> output_filename("o",
                                                  llvm::cl::desc("Output filename"),
                                                  llvm::cl::value_desc("filename"),
                                                  llvm::cl::init("-"));

static llvm::cl::opt<bool>
    split_input_file("split-input-file",
                     llvm::cl::desc("Split the input file into pieces and process each "
                                    "chunk independently"),
                     llvm::cl::init(false));

static llvm::cl::opt<bool>
    verify_diagnostics("verify-diagnostics",
                       llvm::cl::desc("Check that emitted diagnostics match "
                                      "expected-* lines on the corresponding line"),
                       llvm::cl::init(false));

static llvm::cl::opt<bool>
    verify_passes("verify-each",
                  llvm::cl::desc("Run the verifier after each transformation pass"),
                  llvm::cl::init(true));

int main(int argc, char** argv)
{
    llvm::InitLLVM y(argc, argv);
    ngraph::runtime::ngmlir::initializeNGraphMLIR();

    // Register any pass manager command line options.
    mlir::registerPassManagerCLOptions();
    mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
    llvm::cl::ParseCommandLineOptions(argc, argv, "nGraph MLIR modular optimizer driver\n");

    // Set up the input file.
    std::string error_message;
    auto file = mlir::openInputFile(input_filename, &error_message);
    NGRAPH_CHECK(file, error_message);

    auto output = mlir::openOutputFile(output_filename, &error_message);
    NGRAPH_CHECK(output, error_message);

    return failed(mlir::MlirOptMain(output->os(),
                                    std::move(file),
                                    passPipeline,
                                    split_input_file,
                                    verify_diagnostics,
                                    verify_passes));
}
