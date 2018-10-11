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

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph
{
    namespace pass
    {
        class ConstantFolding;
    }
}

class ngraph::pass::ConstantFolding : public ngraph::pass::GraphRewrite
{
    enum class CFTransformations
    {
        RESHAPE,
        BROADCAST,
        PAD,
        DEQUANTIZE
    };

public:
    ConstantFolding()
        : GraphRewrite()
    {
        construct_constant_reshape();
        construct_constant_broadcast();
        construct_constant_pad();
        construct_constant_dequantize();
    }

    //this allows to specify the order in which matchers will be run
    //and also allows to register the same matcher more than once
    ConstantFolding(const std::vector<CFTransformations>& transformations)
        : GraphRewrite()
    {
        for (auto cft : transformations)
        {
            switch (cft)
            {
            case CFTransformations::RESHAPE: construct_constant_reshape(); break;
            case CFTransformations::BROADCAST: construct_constant_broadcast(); break;
            case CFTransformations::PAD: construct_constant_pad(); break;
            case CFTransformations::DEQUANTIZE: construct_constant_dequantize(); break;
            }
        }
    }

private:
    void construct_constant_reshape();
    void construct_constant_broadcast();
    void construct_constant_pad();
    void construct_constant_dequantize();
};
