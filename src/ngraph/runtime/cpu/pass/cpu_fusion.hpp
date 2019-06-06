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
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class CPUFusion;
                class CPUQuantFusion;
            }
        }
    }
}

class CPU_BACKEND_API ngraph::runtime::cpu::pass::CPUFusion : public ngraph::pass::GraphRewrite
{
public:
    typedef ngraph::pass::FusionType FusionType;
    typedef ngraph::pass::FusionTypeMask FusionTypeMask;
    CPUFusion(FusionTypeMask fusions = FusionType::ALL_FUSIONS)
        : GraphRewrite()
    {
        if (fusions.is_set(FusionType::DIFFERENTIABLE_FUSIONS))
        {
            construct_conv_bias(); // DEPRECATED - Use CoreFusion
            construct_sigmoid_multiply();
        }

        if (fusions.is_set(FusionType::REGULAR_FUSIONS))
        {
            construct_matmul();
            construct_matmulbias();
            construct_fprop_bn();
            construct_conv_bias_bprop();
            construct_conv_bias_folded_batch_norm();
            construct_conv_bias_affine_folding();
            construct_groupconv_batchnorm_global_stats_folding();
            construct_groupconv_batchnorm_global_stats_folding_relu();
            construct_batch_norm_relu();
            construct_batch_norm_relu_global_stats();
            construct_conv_relu();
            construct_conv_bias_relu();
            construct_conv_bias_add(); // DEPRECATED - Use CoreFusion
            construct_conv_bias_add_relu();
            construct_leaky_relu();
            construct_bounded_relu();
            // construct_conv_add() should always be after construct_conv_bias()
            construct_conv_add();
            construct_conv_add_relu();
            construct_update_slice();
            construct_fuse_lstm_recurrent_state();
            if (std::getenv("NGRAPH_DECONV_FUSE") != nullptr)
            {
                // Note: enable when the deconv perf is better than convbackpropdata
                construct_deconvolution_affine_folding();
                construct_deconvolution_affine_folding_relu();
            }
        }
    }

private:
    void construct_matmul();
    void construct_matmulbias();
    void construct_conv_bias();
    void construct_conv_bias_bprop();
    void construct_fprop_bn();
    void construct_sigmoid_multiply();
    void construct_batch_norm_relu();
    void construct_batch_norm_relu_global_stats();
    void construct_conv_relu();
    void construct_conv_bias_relu();
    void construct_conv_bias_add();
    void construct_conv_bias_add_relu();
    void construct_conv_add();
    void construct_conv_add_relu();
    void construct_leaky_relu();
    void construct_bounded_relu();
    void construct_conv_bias_folded_batch_norm();
    void construct_conv_bias_affine_folding();
    void construct_groupconv_batchnorm_global_stats_folding();
    void construct_groupconv_batchnorm_global_stats_folding_relu();
    void construct_update_slice();
    void construct_fuse_lstm_recurrent_state();
    void construct_deconvolution_affine_folding();
    void construct_deconvolution_affine_folding_relu();
};

class CPU_BACKEND_API ngraph::runtime::cpu::pass::CPUQuantFusion : public ngraph::pass::GraphRewrite
{
public:
    CPUQuantFusion()
        : GraphRewrite()
    {
        construct_qconv_relu(true);
        construct_qconv_relu(false);
        construct_qavg_pool();
        construct_qmax_pool();
        construct_qconcat();
        construct_qconvb_add();
        construct_dq_q();
        construct_quantized_matmul();
    }

private:
    void construct_qconv_relu(bool with_bias);
    void construct_qavg_pool();
    void construct_qmax_pool();
    void construct_qconcat();
    void construct_dq_q();
    void construct_qconvb_add();
    void construct_quantized_matmul();
};
