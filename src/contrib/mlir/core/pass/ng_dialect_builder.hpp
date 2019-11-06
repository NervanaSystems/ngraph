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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#pragma once

#include "contrib/mlir/core/compiler.hpp"
#include "contrib/mlir/runtime/cpu/memory_manager.hpp"
#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node.hpp"

#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Types.h>
#include <mlir/Pass/Pass.h>

#include <typeindex>
#include <unordered_map>
#include <vector>

using namespace ngraph::runtime::ngmlir;

namespace ngraph
{
    namespace pass
    {
        std::unique_ptr<mlir::Pass>
            createNgDialectConversionPass(const ngraph::op::CompiledKernel* compiledKernel,
                                          mlir::MLIRContext* context);
    }
}
