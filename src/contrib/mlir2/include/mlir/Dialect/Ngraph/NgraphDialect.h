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

#ifndef NGRAPH_NGRAPHDIALECT_H
#define NGRAPH_NGRAPHDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace ngraph {

#include "mlir/Dialect/Ngraph/NgraphOpsDialect.h.inc"

} // namespace ngraph
} // namespace mlir

#endif // NGRAPH_NGRAPHDIALECT_H
