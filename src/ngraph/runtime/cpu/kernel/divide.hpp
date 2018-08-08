/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "ngraph/runtime/cpu/kernel/eigen_thread_pool.hpp"
#include "ngraph/runtime/cpu/tvm_emitter.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {

                DLTensor create_DLTensor(const DLDataType& dtype,
                                         const DLContext& ctx,
                                         const size_t ndim,
                                         tvm_index_t* shape,
                                         void* data)
                {
                    DLTensor t;
                    t.ctx = ctx;
                    t.ndim = ndim;
                    t.dtype = dtype;
                    t.shape = static_cast<int64_t*>(shape);
                    t.strides = nullptr;
                    t.byte_offset = 0;
                    t.data = data;
                    return t;
                }

                template <typename ElementType>
                void divide(void* input0, void* input1, void* output, size_t count)
                {
#if 0
                    Eigen::array<Eigen::Index, 1> out_dims, in_dims;

                    out_dims[0] = in_dims[0] = count;

                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> out(
                        static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> in0(
                        static_cast<ElementType*>(input0), in_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> in1(
                        static_cast<ElementType*>(input1), in_dims);

                    out.device(eigen::global_thread_pool_device) = in0 / in1;
#endif
                    tvm::Var n("n");
                    auto A = tvm::placeholder({n}, tvm::Float(32), "a");
                    auto B = tvm::placeholder({n}, tvm::Float(32), "b");

                    auto C = topi::divide(A, B);

                    auto config = tvm::build_config();
                    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
                    auto target = tvm::target::llvm();

                    auto schedule = topi::x86::default_schedule(target, {C});
                    auto lowered = tvm::lower(schedule, {A, B, C}, "func_divide", binds, config);
                    auto module = tvm::build(lowered, target, tvm::Target(), config);
                    //  std::cout << module->type_key() << std::endl;
                    //  std::cout << module->GetSource() << std::endl;
                    auto func = module->GetFunction("func_divide", false);
                    int ndim = 1;
                    int dtype_code = kDLFloat;
                    int dtype_bits = 32;
                    int dtype_lanes = 1;
                    int device_type = kDLCPU;
                    int device_id = 0;

                    DLDataType dtype;
                    dtype.code = static_cast<uint8_t>(dtype_code);
                    dtype.bits = static_cast<uint8_t>(dtype_bits);
                    dtype.lanes = static_cast<uint16_t>(dtype_lanes);
                    DLContext ctx;
                    ctx.device_type = static_cast<DLDeviceType>(device_type);
                    ctx.device_id = device_id;
                    int64_t dlshape[] = {static_cast<int64_t>(count)};

#if 1
                    DLTensor a = create_DLTensor(dtype, ctx, ndim, dlshape, input0);
                    DLTensor b = create_DLTensor(dtype, ctx, ndim, dlshape, input1);
                    DLTensor c = create_DLTensor(dtype, ctx, ndim, dlshape, output);

                    func(&a,&b,&c);
                    for (int i = 0; i < dlshape[0]; ++i) {
                      std::cout << static_cast<float*>(c.data)[i] << std::endl;
                    }
#endif
#if 0
                    DLTensor* a;
                    DLTensor* b;
                    DLTensor* c;
                    TVMArrayAlloc((tvm_index_t*)dlshape, ndim, dtype_code, dtype_bits, dtype_lanes,
                                  device_type, device_id, &a);
                    TVMArrayAlloc((tvm_index_t*)dlshape, ndim, dtype_code, dtype_bits, dtype_lanes,
                                  device_type, device_id, &b);
                    TVMArrayAlloc((tvm_index_t*)dlshape, ndim, dtype_code, dtype_bits, dtype_lanes,
                                  device_type, device_id, &c);

                    for (int i = 0; i < dlshape[0]; ++i) {
                      static_cast<float*>(a->data)[i] = 1;
                      static_cast<float*>(b->data)[i] = 2;
                      static_cast<float*>(c->data)[i] = 0;
                    }
                    func(a, b, c);

                    for (int i = 0; i < dlshape[0]; ++i) {
                      std::cout << static_cast<float*>(c->data)[i] << std::endl;
                    }
#endif
                }
            }
        }
    }
}
