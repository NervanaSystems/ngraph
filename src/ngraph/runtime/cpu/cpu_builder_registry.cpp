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
#include "ngraph/runtime/cpu/cpu_builder_registry.hpp"

#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            void register_builders()
            {
                register_builders_add_cpp();
                register_builders_allreduce_cpp();
                register_builders_argmax_cpp();
                register_builders_argmin_cpp();
                register_builders_avg_pool_cpp();
                register_builders_batch_norm_cpp();
                register_builders_bounded_relu_cpp();
                register_builders_broadcast_cpp();
                register_builders_broadcast_distributed_cpp();
                register_builders_concat_cpp();
                register_builders_convert_cpp();
                register_builders_convert_layout_cpp();
                register_builders_convolution_cpp();
                register_builders_dot_cpp();
                register_builders_dropout_cpp();
                register_builders_embedding_lookup_cpp();
                register_builders_erf_cpp();
                register_builders_gather_cpp();
                register_builders_gather_nd_cpp();
                register_builders_get_output_element_cpp();
                register_builders_leaky_relu_cpp();
                register_builders_lrn_cpp();
                register_builders_lstm_cpp();
                register_builders_matmul_bias_cpp();
                register_builders_max_cpp();
                register_builders_max_pool_cpp();
                register_builders_min_cpp();
                register_builders_one_hot_cpp();
                register_builders_pad_cpp();
                register_builders_product_cpp();
                register_builders_quantization_cpp();
                register_builders_quantized_avg_pool_cpp();
                register_builders_quantized_conv_cpp();
                register_builders_quantized_dot_cpp();
                register_builders_quantized_matmul_cpp();
                register_builders_reduce_function_cpp();
                register_builders_relu_cpp();
                register_builders_replace_slice_cpp();
                register_builders_reshape_cpp();
                register_builders_reverse_cpp();
                register_builders_reverse_sequence_cpp();
                register_builders_rnn_cpp();
                register_builders_scatter_add_cpp();
                register_builders_scatter_nd_add_cpp();
                register_builders_select_cpp();
                register_builders_state_cpp();
                register_builders_sigmoid_cpp();
                register_builders_slice_cpp();
                register_builders_softmax_cpp();
                register_builders_sum_cpp();
                register_builders_tile_cpp();
                register_builders_topk_cpp();
                register_builders_update_slice_cpp();
                register_builders_quantized_concat_cpp();
                register_builders_quantized_max_pool_cpp();
            }
        }
    }
}
#endif
