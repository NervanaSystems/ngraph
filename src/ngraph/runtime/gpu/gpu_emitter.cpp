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

void runtime::gpu::GPU_Emitter::EmitNop(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAdd(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitDot(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitDivide(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitEqual(const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitGreater(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitGreaterEq(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitLess(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitLessEq(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitLog(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitMaximum(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitMinimum(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitNegative(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitNotEqual(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}
void runtime::gpu::GPU_Emitter::EmitSelect(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSubtract(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitBroadcast(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitConvert(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitConstant(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReshape(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitFunctionCall(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReduce(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSign(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSlice(const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSum(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitExp(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSin(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSinh(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitCos(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitCosh(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitTan(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitTanh(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAsin(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAcos(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitAtan(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitPower(const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReplaceSlice(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitOneHot(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitCeiling(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitFloor(const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitSqrt(const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitConvolution(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitNot(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitMaxPool(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReverse(const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

//------------------------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------------------------

void runtime::gpu::GPU_Emitter::generate_call(
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out,
    shared_ptr<Function> function)
{
    vector<string> input_names;
    vector<string> output_names;

    for (const runtime::gpu::GPU_TensorViewWrapper& input : args)
    {
        input_names.push_back(input.get_name());
    }

    for (const runtime::gpu::GPU_TensorViewWrapper& output : out)
    {
        output_names.push_back(output.get_name());
    }

    m_out << "void* args[] =\n{";
    m_out.indent++;
    m_out << "\n" << join(input_names, ",\n");
    m_out.indent--;
    m_out << "\n};\n";

    m_out << "void* out[] =\n{";
    m_out.indent++;
    m_out << "\n" << join(output_names, ",\n");
    m_out.indent--;
    m_out << "\n};\n";

    m_out << "\n";
    m_out << function->get_name() << "(args, out);\n";
}

static string format_name(const string& name)
{
    string rc;
    if (!name.empty())
    {
        rc = " " + name;
    }
    return rc;
}

void runtime::gpu::GPU_Emitter::EmitAbs(const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitConcat(const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitMultiply(
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}
