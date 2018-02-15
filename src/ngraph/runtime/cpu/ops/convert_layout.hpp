/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/ops/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class LayoutDescriptor;

            namespace op
            {
                /// \brief Layout Conversion
                ///
                /// Converts an input tensor to a tensor with the given layout descriptor
                class ConvertLayout : public ngraph::op::util::RequiresTensorViewArgs
                {
                public:
                    ConvertLayout(
                        const std::shared_ptr<Node>& arg,
                        const std::shared_ptr<ngraph::runtime::cpu::LayoutDescriptor>& layout);

                    ConvertLayout(
                        const std::shared_ptr<Node>& arg,
                        size_t output_index,
                        const std::shared_ptr<ngraph::runtime::cpu::LayoutDescriptor>& layout);

                    virtual std::shared_ptr<Node>
                        copy_with_new_args(const Nodes& new_args) const override
                    {
                        if (new_args.size() != 1)
                        {
                            throw ngraph_error("Incorrect number of new arguments");
                        }
                        return std::make_shared<ConvertLayout>(new_args.at(0), output_layout);
                    }

                protected:
                    size_t arg_output_index;
                    std::shared_ptr<ngraph::runtime::cpu::LayoutDescriptor> output_layout;
                };
            }
        }
    }
}
