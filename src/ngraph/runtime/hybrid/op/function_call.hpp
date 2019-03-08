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

#include "ngraph/op/op.hpp"
#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace hybrid
        {
            namespace op
            {
                class FunctionCall;
            }
        }
    }
}

class ngraph::runtime::hybrid::op::FunctionCall : public ngraph::op::Op
{
public:
    FunctionCall(const NodeVector& outputs,
                 const NodeVector& inputs,
                 std::shared_ptr<Function> function,
                 std::shared_ptr<Backend> backend);

    std::shared_ptr<Backend> get_backend() const;
    std::shared_ptr<Executable> get_executable() const;
    std::shared_ptr<Function> get_function() const;

private:
    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    const NodeVector m_function_outputs;
    std::shared_ptr<Function> m_function;
    std::shared_ptr<Backend> m_backend;
    std::shared_ptr<Executable> m_executable;
};
