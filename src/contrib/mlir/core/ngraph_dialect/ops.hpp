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

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.

#pragma once

#include <cstdarg>
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"

// attributes
// Currently table-gen dictates that enum attributes are in global namespace
#include "ops_attributes.h.inc"

namespace mlir
{
// interfaces
#include "ops_interfaces.h.inc"

// ops
#define GET_OP_CLASSES
#include "ops.h.inc"
#undef GET_OP_CLASSES
}

void setBufferId(mlir::Operation* op, mlir::IntegerAttr attr);
mlir::IntegerAttr setBufferId(mlir::Operation* op, unsigned val);
mlir::IntegerAttr getBufferId(mlir::Operation* op);
