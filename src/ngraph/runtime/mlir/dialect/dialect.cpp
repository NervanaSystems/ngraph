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

#include "dialect.hpp"
#include "ops.hpp"
#include "type.hpp"
namespace ngraph
{
    using namespace runtime::cpu;

    /// Register a dialect and its types
    /// Usage:
    /// mlir::registerDialect<ngraph::runtime::cpu::ngdialect::Dialect>();
    NGDialect::NGDialect(mlir::MLIRContext* ctx)
        : mlir::Dialect("ng", ctx)
    {
        addTypes<NGTensorType>();
        addOperations<NG_AddOp>();
        addOperations<NG_MatmulBiasOp>();
        addOperations<NG_ReturnOp>();
        addOperations<NG_FakeInput>();
    }

    void NGDialect::printType(mlir::Type type, raw_ostream& os) const
    {
        auto arrayTy = type.dyn_cast<NGTensorType>();
        if (!arrayTy)
        {
            NGRAPH_ASSERT(0) << "Incorrect type to print?";
        }
        os << "tensor";
        if (!arrayTy.getShape().empty())
        {
            os << "<";
            mlir::interleaveComma(arrayTy.getShape(), os);
            os << ">";
        }
    }
}
