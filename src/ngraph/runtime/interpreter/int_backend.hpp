//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/interpreter/node_wrapper.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/state/rng_state.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/runtime/reference/allreduce.hpp"
#endif

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class INTBackend;
        }
    }
}

class ngraph::runtime::interpreter::INTBackend : public Backend
{
public:
    INTBackend();
    INTBackend(const std::vector<std::string>& unsupported_op_name_list);

    std::shared_ptr<Tensor>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<Tensor> create_tensor(const element::Type& type, const Shape& shape) override;

    Handle compile(std::shared_ptr<Function> function,
                   bool enable_performance_collection = false) override;

    bool is_supported(const Node& node) const override;

private:
    std::set<std::string> m_unsupported_op_name_list;
};
