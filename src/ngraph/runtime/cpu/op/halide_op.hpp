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

#include <list>
#include <vector>

#include "ngraph/op/op.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace op
            {
                class HalideOp : public ngraph::op::Op
                {
                public:
                    CPU_BACKEND_API
                    static constexpr NodeTypeInfo type_info{"HalideOp", 0};
                    const NodeTypeInfo& get_type_info() const override { return type_info; }
                    HalideOp(const OutputVector& args,
                             const std::list<Output<Node>>& ops,
                             const element::Type& out_type,
                             const Shape& out_shape);

                    virtual void validate_and_infer_types() override;

                    virtual std::shared_ptr<Node>
                        copy_with_new_args(const NodeVector& new_args) const override;

                    const std::list<Output<Node>>& get_ops() const { return m_ops; }
                private:
                    std::list<Output<Node>> m_ops;
                    element::Type m_output_type;
                    Shape m_output_shape;
                };
            }
        }
    }
}
