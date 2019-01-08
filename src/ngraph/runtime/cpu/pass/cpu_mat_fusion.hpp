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

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class CPURnnMatFusion : public ngraph::pass::FunctionPass
                {
                public:
                    virtual bool
                        run_on_function(std::shared_ptr<ngraph::Function> function) override;
                };
                class CPUBatchFusion : public ngraph::pass::FunctionPass
                {
                public:
                    enum FusionType
                    {
                        //`DIFFERENTIABLE_FUSIONS` produce ops that support autodiff
                        // i.e. implement `generate_adjoints`
                        DIFFERENTIABLE_FUSIONS = 0x1,
                        REGULAR_FUSIONS = 0x2,
                        ALL = 0xFFFFFFFF
                    };

                    CPUBatchFusion(FusionType type = ALL)
                        : m_fusion_type(type)
                        , FunctionPass()
                    {
                    }
                    virtual bool
                        run_on_function(std::shared_ptr<ngraph::Function> function) override;

                private:
                    FusionType m_fusion_type;
                };
            }
        }
    }
}
