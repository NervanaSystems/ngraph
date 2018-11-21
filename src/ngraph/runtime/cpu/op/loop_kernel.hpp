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

#include "ngraph/op/op.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace op
            {
                /// \brief LoopKernel represents graphs consisting
                /// of arithmetic operations that can be executed in the same loop
                class LoopKernel : public ngraph::op::Op
                {
                public:
                    LoopKernel(const NodeVector& node_list,
                               const NodeVector& outputs,
                               const NodeVector& args);
                    virtual std::shared_ptr<Node>
                        copy_with_new_args(const NodeVector& new_args) const override;

                    const NodeVector& get_node_list() const { return m_node_list; }
                    const NodeVector& get_kernel_outputs() const { return m_output_nodes; }
                private:
                    NodeVector m_node_list;
                    NodeVector m_output_nodes;
                };
            }
        }
    }
}
