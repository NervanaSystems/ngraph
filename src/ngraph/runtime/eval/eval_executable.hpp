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

#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/log.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/executable.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/opt_kernel/broadcast.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace eval
        {
            class EVALBackend;
            class EVALExecutable;

            // This expands the op list in op_tbl.hpp into a list of enumerations that look like
            // this:
            // Abs,
            // Acos,
            // ...
            enum class OP_TYPEID
            {
#define NGRAPH_OP(NAME, VERSION) NAME##_v##VERSION,
#include "ngraph/op_version_tbl.hpp"
#undef NGRAPH_OP
                UnknownOp
            };
        }
    }
}

class ngraph::runtime::eval::EVALExecutable : public runtime::Executable
{
    friend class EVALBackend;

public:
    EVALExecutable(const std::shared_ptr<Function>& function,
                   bool enable_performance_collection = false);

    bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& intputs) override;

private:
    std::shared_ptr<Function> m_function;
    NodeVector m_nodes;
    static OP_TYPEID get_typeid(const Node& node);
};
