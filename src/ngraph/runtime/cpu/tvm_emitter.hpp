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

#include <map>
#include <dlpack/dlpack.h>
#include <tvm/tvm.h>

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace tvm {
                const DLDataType DLType_Float32 {kDLFloat, 32, 1};
            }
            class TVMEmitter
            {
            public:
                TVMEmitter();
                ~TVMEmitter();
                enum class TVM_OP : uint32
                {
                    Divide
                }


            private:
                void build_divide();

                DLTensor create_dltensor(const size_t ndim,
                                         tvm_index_t* shape,
                                         const DLDataType type,
                                         void* data);


            private:
                std::map<TVM_OP, tvm::PackedFunc> m_op_functors;
                DLContext m_dl_ctx;
            };

            namespace tvm {
                template <typename ElementType>
                void divide(void* input0, void* input1, void* output, size_t count)
                {
                  int ndim = 1;
                  int64_t shape[] = {static_cast<int64_t>(count)};
    #if 1
                  DLTensor a = create_dltensor(dtype, ctx, ndim, shape, input0);
                  DLTensor b = create_dltensor(dtype, ctx, ndim, shape, input1);
                  DLTensor c = create_dltensor(dtype, ctx, ndim, shape, output);

                  func(&a,&b,&c);
                  for (int i = 0; i < dlshape[0]; ++i) {
                    std::cout << static_cast<float*>(c.data)[i] << std::endl;
                  }
    #endif
                }
            }
        }
    }
}
