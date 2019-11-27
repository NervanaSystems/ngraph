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
        bool revalidate_and_ensure_static(std::shared_ptr<ngraph::Node> n);
    }
}

class NGRAPH_API ngraph::pass::ConstantFolding : public ngraph::pass::GraphRewrite
{
public:
    enum class CFTransformations
    {
        RESHAPE,
        BROADCAST,
        DYN_BROADCAST,
        PAD,
        DEQUANTIZE,
        UNARY,
        BINARY,
        QUANTIZE,
        CONVERT,
        SHAPE_OF,
        REVERSE,
        ARITHMETIC_REDUCTION,
        LOGICAL_REDUCTION,
        CONCAT,
        GATHER,
        SLICE,
        DYN_SLICE,
        STRIDED_SLICE,
        DYN_RESHAPE,
        TRANSPOSE,
        RANGE,
        SELECT,
        SQUEEZE,
        UNSQUEEZE
    };

    ConstantFolding(const ngraph::BuildNodeExecutorMap& cfmap = ngraph::BuildNodeExecutorMap())
        : GraphRewrite()
    {
        m_cfmap = cfmap;
        m_enable_shape_inference = true;

        construct_constant_reshape();
        construct_constant_broadcast();
        construct_constant_dyn_broadcast();
        construct_constant_pad();
        construct_constant_unary();
        construct_constant_binary();
        construct_constant_quantize();
        construct_constant_dequantize();
        construct_constant_convert();
        construct_constant_shape_of();
        construct_constant_reverse();
        construct_constant_arithmetic_reduction();
        construct_constant_logical_reduction();
        construct_constant_concat();
        construct_constant_gather();
        construct_constant_slice();
        construct_constant_dyn_slice();
        construct_constant_strided_slice();
        construct_constant_dyn_reshape();
        construct_constant_transpose();
        construct_constant_range();
        construct_constant_select();
        construct_constant_squeeze();
        construct_constant_unsqueeze();
    }

    // this allows to specify the order in which matchers will be run
    // and also allows to register the same matcher more than once
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
            case CFTransformations::DYN_BROADCAST: construct_constant_dyn_broadcast(); break;
            case CFTransformations::PAD: construct_constant_pad(); break;
            case CFTransformations::UNARY: construct_constant_unary(); break;
            case CFTransformations::BINARY: construct_constant_binary(); break;
            case CFTransformations::DEQUANTIZE: construct_constant_dequantize(); break;
            case CFTransformations::QUANTIZE: construct_constant_quantize(); break;
            case CFTransformations::CONVERT: construct_constant_convert(); break;
            case CFTransformations::SHAPE_OF: construct_constant_shape_of(); break;
            case CFTransformations::REVERSE: construct_constant_reverse(); break;
            case CFTransformations::ARITHMETIC_REDUCTION:
                construct_constant_arithmetic_reduction();
                break;
            case CFTransformations::LOGICAL_REDUCTION:
                construct_constant_logical_reduction();
                break;
            case CFTransformations::CONCAT: construct_constant_concat(); break;
            case CFTransformations::GATHER: construct_constant_gather(); break;
            case CFTransformations::SLICE: construct_constant_slice(); break;
            case CFTransformations::DYN_SLICE: construct_constant_dyn_slice(); break;
            case CFTransformations::STRIDED_SLICE: construct_constant_strided_slice(); break;
            case CFTransformations::DYN_RESHAPE: construct_constant_dyn_reshape(); break;
            case CFTransformations::TRANSPOSE: construct_constant_transpose(); break;
            case CFTransformations::RANGE: construct_constant_range(); break;
            case CFTransformations::SELECT: construct_constant_select(); break;
            case CFTransformations::SQUEEZE: construct_constant_squeeze(); break;
            case CFTransformations::UNSQUEEZE: construct_constant_unsqueeze(); break;
            }
        }
    }

private:
    void construct_constant_reshape();
    void construct_constant_broadcast();
    void construct_constant_dyn_broadcast();
    void construct_constant_pad();
    void construct_constant_unary();
    void construct_constant_binary();
    void construct_constant_quantize();
    void construct_constant_dequantize();
    void construct_constant_convert();
    void construct_constant_shape_of();
    void construct_constant_reverse();
    void construct_constant_arithmetic_reduction();
    void construct_constant_logical_reduction();
    void construct_constant_concat();
    void construct_constant_gather();
    void construct_constant_slice();
    void construct_constant_dyn_slice();
    void construct_constant_strided_slice();
    void construct_constant_dyn_reshape();
    void construct_constant_transpose();
    void construct_constant_range();
    void construct_constant_select();
    void construct_constant_squeeze();
    void construct_constant_unsqueeze();

    ngraph::BuildNodeExecutorMap m_cfmap;
};
