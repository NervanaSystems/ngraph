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

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.

#include "fusion_dialect.hpp"
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Parser.h>
#include "ngraph/check.hpp"
#include "ops.hpp"
#include "type.hpp"

using namespace mlir;

NGraphFusedOpsDialect::NGraphFusedOpsDialect(mlir::MLIRContext* ctx)
    : mlir::Dialect(getDialectNamespace(), ctx)
{
    // TODO (pruthvi) add nGFusedDialect Tenosr types and element types
    // addTypes<NGFusedTensorType>
    // ....

    // add all the supported ops here addOperations<...>()
}
