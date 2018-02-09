// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------
#include <algorithm>
#include <map>

#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn_v7.h>

#include "ngraph/node.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/util.hpp"



#define NVRTC_SAFE_CALL(x) \ 
do { \ 
    nvrtcResult result = x; \ 
        if (result != NVRTC_SUCCESS) { \ 
            std::cerr << "\nerror: " #x " failed with error " \ 
                << nvrtcGetErrorString(result) << '\n'; \
                exit(1); \ 
        } \
} while(0) 

#define CUDA_SAFE_CALL(x) \ 
do { \ 
    CUresult result = x; \ 
        if (result != CUDA_SUCCESS) { \ 
            const char *msg; \ 
                cuGetErrorName(result, &msg); \ 
                std::cerr << "\nerror: " #x " failed with error " \ 
                << msg << '\n'; \ 
                exit(1); \ 
        } \ 
} while(0)

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace cuda
            {
                namespace kernel
                {

                    void emit_abs(void* in, void* out, size_t count)
                    {
                        void *argsList[] = {In, Out, &count};
                        CUDA_SAFE_CALL(
                        cuLaunchKernel(cuda_op_abs_kernel, 
                        count , 1, 1, // grid dim 
                        1, 1, 1, // block dim 
                        0, NULL, // shared mem and stream 
                        argsList, 0)); // arguments 
                        CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output. 
                    }
                    void emit_broadcast(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const AxisSet& broadcast_axes)
                    {
                    }

                    //
                    // For the reference kernel this is gpud on, see ngraph/runtime/kernel/concat.hpp.
                    //
                    void emit_concat(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::vector<std::string>& args,
                            const std::string& out,
                            const std::vector<Shape>& in_shapes,
                            const Shape& out_shape,
                            size_t concatenation_axis)
                    {
                    }

                    void emit_replace_slice(
                            codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& arg1, // replacement value
                            const std::string& out,
                            const Shape& arg1_shape,
                            const Shape& out_shape,
                            const Coordinate& lower_bounds,
                            const Coordinate& upper_bounds,
                            const Strides& strides)
                    {
                    }

                    void emit_slice(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const Coordinate& lower_bounds,
                            const Coordinate& upper_bounds,
                            const Strides& strides)
                    {
                    }

                    void emit_reshape(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const AxisVector& arg0_axis_order)
                    {
                    }

                    void emit_sum(codegen::CodeWriter& writer,
                            const std::string& element_type,
                            const std::string& arg0, // replacement context
                            const std::string& out,
                            const Shape& arg0_shape,
                            const Shape& out_shape,
                            const AxisSet& reduction_axes)
                    {
                    }

                }
            }
        }
    }
}
