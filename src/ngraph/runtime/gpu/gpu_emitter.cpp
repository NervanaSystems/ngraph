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
#include <cmath>
#include <iostream>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

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
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void runtime::gpu::GPU_Emitter::EmitNop(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAbs(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAdd(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitConcat(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitDot(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    const Shape& arg0_shape = args[0].get_shape();
    const Shape& arg1_shape = args[1].get_shape();
    if (arg0_shape.empty() || arg1_shape.empty())
    {
        auto& first = (arg0_shape.empty() ? args[0] : args[1]);
        auto& second = (arg0_shape.empty() ? args[1] : args[0]);
    }

    // clang-format off
    else if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1))
    {
      // TODO Assert arg0_shape[0] == arg1_shape[0]?
      writer << "{   // " << n->get_name() << "\n";
      writer.indent++;
      writer << "cublas::cublasSdot("
          << "cublas_handle,"
          << arg0_shape[0] << ","
          << args[0].get_name() << ","
          // Todo handle striding?
          << "1,"
          << args[1].get_name() << ","
          << "1,"
          << out[0].get_name() << ")";
      writer.indent--;
      writer << "}\n";
    }
    // clang-format on
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1))
    {
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2))
    {
        // GEMM Call
    }
    else
    {
        // General ND Call?
    }
}

void runtime::gpu::GPU_Emitter::EmitDivide(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitEqual(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitGreater(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitGreaterEq(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitLess(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitLessEq(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitLog(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitMaximum(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitMinimum(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitNegative(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitNotEqual(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}
void runtime::gpu::GPU_Emitter::EmitSelect(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSubtract(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitBroadcast(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitConvert(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitConstant(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReshape(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitFunctionCall(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReduce(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSign(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSlice(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSum(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitMultiply(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitExp(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSin(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSinh(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitCos(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitCosh(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitTan(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitTanh(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAsin(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAcos(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAtan(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitPower(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReplaceSlice(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitOneHot(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitCeiling(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitFloor(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSqrt(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitConvolution(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitNot(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitMaxPool(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReverse(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReduceWindow(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSelectAndScatter(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}
