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

#include <CPP/custom_gpu_primitive.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

#include "ngraph/node.hpp"

using namespace std;
using namespace ngraph;

void runtime::intelgpu::CustomKernels::queue_krnl(const krnl_info& krnl_info,
                                                  const shared_ptr<Node>& op)
{
    for (const auto& kr : krnl_info)
    {
        const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(kr.m_type, kr.m_shape);

        const cldnn::custom_gpu_primitive kernel_item(kr.m_name,
                                                      kr.m_inputs,
                                                      {kr.m_code},
                                                      kr.m_entry_point,
                                                      get_kernel_args(kr.m_inputs.size(), 1),
                                                      "",
                                                      layout,
                                                      kr.m_gws,
                                                      kr.m_lws);
        stream.add(kernel_item);
    }
}

void runtime::intelgpu::arguments_check(const shared_ptr<Node>& op, size_t input, size_t output)
{
    if (op->get_input_size() != input || op->get_output_size() != output)
    {
        ostringstream os;
        os << "Operation \"" << op->description() << "\" input and output sizes mismatch."
           << " Expected input size=" << input << ", provided=" << op->get_input_size()
           << ". Expected output size=" << output << ", provided=" << op->get_output_size();
        throw invalid_argument(os.str());
    }
}
