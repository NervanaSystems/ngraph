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
#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "ngraph/check.hpp"
namespace mlir
{
    class NGraphOpsDialect : public mlir::Dialect
    {
    public:
        explicit NGraphOpsDialect(mlir::MLIRContext* ctx);
        mlir::Type parseType(llvm::StringRef tyData, mlir::Location loc) const override
        {
            NGRAPH_CHECK(false, "Unsupported type parsing.");
            return mlir::Type();
        }
        void printType(mlir::Type type, llvm::raw_ostream& os) const override;

        static StringRef getDialectNamespace() { return "ng"; }
    };
}
