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

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace pass
    {
        class ConstantFolding;
    }
}

class ngraph::pass::ConstantFolding : public ngraph::pass::GraphRewrite
{
public:
    enum class CFTransformations
    {
        RESHAPE,
        BROADCAST,
        PAD,
        DEQUANTIZE,
        UNARY,
        BINARY,
        QUANTIZE,
        CONVERT,
        SHAPE_OF,
        REVERSE,
        PRODUCT,
        SUM,
        CONCAT,
        GATHER
    };

    ConstantFolding(const ngraph::BuildNodeExecutorMap& cfmap = ngraph::BuildNodeExecutorMap())
        : GraphRewrite()
    {
        m_cfmap = cfmap;
        construct_constant_reshape();
        construct_constant_broadcast();
        construct_constant_pad();
        construct_constant_unary();
        construct_constant_binary();
        construct_constant_quantize();
        construct_constant_dequantize();
        construct_constant_convert();
        construct_constant_shape_of();
        construct_constant_reverse();
        construct_constant_product();
        construct_constant_sum();
        construct_constant_concat();
        construct_constant_gather();
    }

    //this allows to specify the order in which matchers will be run
    //and also allows to register the same matcher more than once
    ConstantFolding(const std::vector<CFTransformations>& transformations,
                    const ngraph::BuildNodeExecutorMap& cfmap = ngraph::BuildNodeExecutorMap())
        : GraphRewrite()
    {
        m_cfmap = cfmap;
        for (auto cft : transformations)
        {
            switch (cft)
            {
            case CFTransformations::RESHAPE: construct_constant_reshape(); break;
            case CFTransformations::BROADCAST: construct_constant_broadcast(); break;
            case CFTransformations::PAD: construct_constant_pad(); break;
            case CFTransformations::UNARY: construct_constant_unary(); break;
            case CFTransformations::BINARY: construct_constant_binary(); break;
            case CFTransformations::DEQUANTIZE: construct_constant_dequantize(); break;
            case CFTransformations::QUANTIZE: construct_constant_quantize(); break;
            case CFTransformations::CONVERT: construct_constant_convert(); break;
            case CFTransformations::SHAPE_OF: construct_constant_shape_of(); break;
            case CFTransformations::REVERSE: construct_constant_reverse(); break;
            case CFTransformations::PRODUCT: construct_constant_product(); break;
            case CFTransformations::SUM: construct_constant_sum(); break;
            case CFTransformations::CONCAT: construct_constant_concat(); break;
            case CFTransformations::GATHER: construct_constant_gather(); break;
            }
        }
    }

private:
    void construct_constant_reshape();
    void construct_constant_broadcast();
    void construct_constant_pad();
    void construct_constant_unary();
    void construct_constant_binary();
    void construct_constant_quantize();
    void construct_constant_dequantize();
    void construct_constant_convert();
    void construct_constant_shape_of();
    void construct_constant_reverse();
    void construct_constant_product();
    void construct_constant_sum();
    void construct_constant_concat();
    void construct_constant_gather();

    ngraph::BuildNodeExecutorMap m_cfmap;
};
