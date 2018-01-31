// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "ngraph/descriptor/output.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        namespace cpu
        {
            class CPUTensorAllocation : public ngraph::pass::FunctionPass
            {
            public:
                virtual bool run_on_function(std::shared_ptr<ngraph::Function> function) override
                {
                    for (std::shared_ptr<ngraph::Node> node : function->get_ops())
                    {
                        for (size_t i = 0; i < node->get_output_size(); ++i)
                        {
                            auto tensor_view = node->get_output_tensor_view(i);
                            auto cpu_tensor_view = std::static_pointer_cast<ngraph::runtime::cpu::CPUTensorView>(tensor_view);

                        }
                    }
                    return false;
                }
            };
        }
    }
}
