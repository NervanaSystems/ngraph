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
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class CPUFusion;
            }
        }
    }
}

class ngraph::runtime::cpu::pass::CPUFusion : public ngraph::pass::GraphRewrite
{
public:
    // 30 different fusion groups that we can nest/mix&match/etc
    // should be good enough for quite a while
    enum fusions
    {
        //`DIFFERENTIABLE_FUSIONS` produce ops that support autodiff
        // i.e. implement `generate_adjoints`
        DIFFERENTIABLE_FUSIONS = 0x1,
        REGULAR_FUSIONS = 0x2,
        ALL = 0xFFFFFFFF
    };

    CPUFusion(int fusions = ALL)
        : GraphRewrite()
    {
        if (fusions & REGULAR_FUSIONS)
        {
            construct_matmul();
            construct_matmulbias();
            construct_fprop_bn();
            construct_zero_padded_reshaped_conv();
            construct_zero_padded_conv();
            construct_zero_padded_conv_backprop_filters();
            construct_conv_bias_bprop();
            construct_batch_norm_relu();
            construct_batch_norm_relu_global_stats();
            construct_conv_relu();
            construct_conv_bias_relu();
            construct_conv_bias_add();
            construct_conv_bias_add_relu();
            construct_bounded_relu();
        }

        if (fusions & DIFFERENTIABLE_FUSIONS)
        {
            construct_conv_bias();
            construct_sigmoid_multiply();
        }
    }

private:
    void construct_matmul();
    void construct_matmulbias();
    void construct_conv_bias();
    void construct_conv_bias_bprop();
    void construct_fprop_bn();
    void construct_sigmoid_multiply();
    void construct_zero_padded_reshaped_conv();
    void construct_zero_padded_conv();
    void construct_zero_padded_conv_backprop_filters();
    void construct_batch_norm_relu();
    void construct_batch_norm_relu_global_stats();
    void construct_conv_relu();
    void construct_conv_bias_relu();
    void construct_conv_bias_add();
    void construct_conv_bias_add_relu();
    void construct_bounded_relu();
};
