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

#include "ngraph/pass/graph_rewrite.hpp"

namespace ngraph
{
    namespace pass
    {
        class CoreFusion;
    }
}

class ngraph::pass::CoreFusion : public ngraph::pass::GraphRewrite
{
public:
    CoreFusion()
        : GraphRewrite()
    {
        construct_relu();
        construct_folded_batch_norm();
        construct_conv_affine_folding();
        construct_sigmoid();
        construct_sigmoid_bprop();
        construct_optimized_strided_conv();
    }
    void construct_relu();
    void construct_folded_batch_norm();
    void construct_conv_affine_folding();
    void construct_sigmoid();
    void construct_sigmoid_bprop();
    void construct_optimized_strided_conv();
};
