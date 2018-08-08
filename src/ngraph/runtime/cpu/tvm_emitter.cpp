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
#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/build_module.h>
#include <topi/broadcast.h>
#include <topi/x86/default.h>

#include "tvm_emitter.hpp"

using namespace ngraph::runtime::cpu;

TVMEmitter::TVMEmitter()
{
    // TODO lfeng: hard coded to float32
    m_dl_dtype.code = static_cast<uint8_t>(kDLFloat);
    m_dl_dtype.bits = static_cast<uint8_t>(32);
    m_dl_dtype.lanes = static_cast<uint16_t>(1);
    m_dl_ctx.device_type = static_cast<DLDeviceType>(kDLCPU);
    m_dl_ctx.device_id = 0;
}
TVMEmitter::~TVMEmitter()
{

}

TVMEmitter::build_divide()
{
  const std::string op_name = "func_divide";
  tvm::Var n("n");
  auto A = tvm::placeholder({n}, tvm::Float(32), "a");
  auto B = tvm::placeholder({n}, tvm::Float(32), "b");

  auto C = topi::divide(A, B);

  auto config = tvm::build_config();
  std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
  auto target = tvm::target::llvm();

  auto schedule = topi::x86::default_schedule(target, {C});
  auto lowered = tvm::lower(schedule, {A, B, C}, op_name, binds, config);
  auto module = tvm::build(lowered, target, tvm::Target(), config);
  m_op_functors.push_back(TVM_OP::Divide, module->GetFunction(op_name));
}
