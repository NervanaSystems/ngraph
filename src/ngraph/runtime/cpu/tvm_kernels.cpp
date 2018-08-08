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

#include <memory>
#include <string>

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <topi/broadcast.h>
#include <topi/x86/default.h>
#include <tvm/build_module.h>
#include <tvm/operation.h>
#include <tvm/tvm.h>

#include "tvm_kernels.hpp"

using namespace ngraph::runtime::cpu;

TVMInstance::TVMInstance()
{
    // TODO lfeng: hard coded to float32
    m_config = tvm::build_config();
    m_target = tvm::target::llvm();
    m_dl_ctx.device_type = static_cast<DLDeviceType>(kDLCPU);
    m_dl_ctx.device_id = 0;
}
TVMInstance::~TVMInstance()
{
}
DLTensor TVMInstance::create_dltensor(const DLDataType& type,
                                      const size_t ndim,
                                      tvm_index_t* shape,
                                      void* data)
{
    DLTensor t;
    t.ctx = m_dl_ctx;
    t.ndim = ndim;
    t.dtype = type;
    t.shape = static_cast<int64_t*>(shape);
    t.strides = nullptr;
    t.byte_offset = 0;
    t.data = data;
    return t;
}
static const DLDataType DLType_Float32{kDLFloat, 32, 1};

template <>
tvm::PackedFunc tvm_kernel::build_divide<float>(const std::unique_ptr<TVMInstance>& tvm_instance)
{
    std::cout << "divide float build" << std::endl;
    tvm::Var n("n");
    auto A = tvm::placeholder({n}, tvm::Float(32), "a");
    auto B = tvm::placeholder({n}, tvm::Float(32), "b");

    auto C = topi::divide(A, B);

    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;

    auto schedule = topi::x86::default_schedule(tvm_instance->target(), {C});
    auto lowered = tvm::lower(schedule, {A, B, C}, "func_divide", binds, tvm_instance->config());
    auto module =
        tvm::build(lowered, tvm_instance->target(), tvm::Target(), tvm_instance->config());
    // store module to keep its lifetime
    tvm_instance->add_module(module);
    return module->GetFunction("func_divide", false);
}

template <>
void tvm_kernel::binary_elemwise_compute<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                                const tvm::PackedFunc& func,
                                                void* input0,
                                                void* input1,
                                                void* output,
                                                size_t count)
{
    std::cout << "divide float compute" << std::endl;
    int64_t dlshape[] = {static_cast<int64_t>(count)};
    DLTensor a = tvm_instance->create_dltensor(DLType_Float32, 1, dlshape, input0);
    DLTensor b = tvm_instance->create_dltensor(DLType_Float32, 1, dlshape, input1);
    DLTensor c = tvm_instance->create_dltensor(DLType_Float32, 1, dlshape, output);

    func(&a, &b, &c);
}
