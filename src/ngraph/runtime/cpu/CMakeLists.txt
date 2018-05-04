# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

message(STATUS "****************** CPU backend enabled")

# Add sources for the CPU backend
# and all its dependencies
set(SRC
    cpu_call_frame.cpp
    cpu_backend.cpp
    cpu_kernels.cpp
    cpu_kernel_emitters.cpp
    cpu_kernel_utils.cpp
    cpu_emitter.cpp
    cpu_external_function.cpp
    cpu_tensor_view.cpp
    cpu_tensor_view_wrapper.cpp
    cpu_layout_descriptor.cpp
    cpu_tracing.cpp
    mkldnn_emitter.cpp
    mkldnn_invoke.cpp
    mkldnn_utils.cpp
    kernel/eigen_thread_pool.cpp
    kernel/pad.cpp
    kernel/reduce_max.cpp
    kernel/reduce_sum.cpp
    kernel/reshape.cpp
    op/conv_bias.cpp
    op/conv_relu.cpp
    op/convert_layout.cpp
    op/sigmoid.cpp
    op/matmul_bias.cpp
    op/batch_norm_relu.cpp
    pass/cpu_assignment.cpp
    pass/cpu_fusion.cpp
    pass/cpu_layout.cpp
    pass/cpu_nop_elimination.cpp
    pass/cpu_rnn_mat_fusion.cpp
    pass/cpu_post_layout_optimizations.cpp
)

add_library(cpu_backend SHARED ${SRC})
add_dependencies(cpu_backend codegen libmkldnn libjson)
target_link_libraries(cpu_backend PUBLIC codegen libmkldnn libjson)

