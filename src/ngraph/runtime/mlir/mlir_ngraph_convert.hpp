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

#pragma once

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Translation.h"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/function.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace mlir
        {
            class NgraphToMlir;
        }
    }
}

class ngraph::runtime::mlir::NgraphToMlir
{
public:
    static void convert_function(const ngraph::Function* function);

private:
    NgraphToMlir(::mlir::MLIRContext* context);
    void convert(const ngraph::Function* function);
    // ::mlir::Type get_mlir_type(const descriptor::Tensor* tensor);
    ::mlir::Type get_mlir_type(const element::Type& type);
    // llvm::SmallVectorImpl<int64_t> get_mlir_shape(const ngraph::Shape& ngraph_shape);

    ::mlir::MLIRContext* m_context;
};
