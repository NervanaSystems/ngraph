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
    CPUFusion()
        : GraphRewrite()
    {
        construct_matmul();
        construct_matmulbias();
        construct_fprop_bn();
        construct_zero_padded_reshaped_conv();
        construct_zero_padded_conv();
        construct_sigmoid();
        construct_sigmoid_bprop();
        construct_conv_bias();
        construct_batch_norm_relu();
        construct_conv_relu();
        construct_lstm_fprop();
    }

private:
    void construct_matmul();
    void construct_matmulbias();
    void construct_conv_bias();
    void construct_fprop_bn();
    void construct_sigmoid();
    void construct_sigmoid_bprop();
    void construct_zero_padded_reshaped_conv();
    void construct_zero_padded_conv();
    void construct_batch_norm_relu();
    void construct_conv_relu();
    void construct_lstm_fprop();
};
