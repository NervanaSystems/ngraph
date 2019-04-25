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
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void runtime::intelgpu::CustomKernels::queue_krnl(const krnl_info& krnl_info,
                                                  const shared_ptr<Node>& op)
{
    for (const CustomKernelInfo& kr : krnl_info)
    {
// Need to save this code to allow further work on it later
#if 0
        mkldnn::engine eng(0);
        shared_ptr<mkldnn::stream> mkldnn_stream = make_shared<mkldnn::stream>(eng);
        cl_device_id device = eng.get_ocl_device();

        const char* source_code = kr.m_code.c_str();
        const size_t source_code_length = strlen(source_code);
        cl_int errcode = CL_SUCCESS;

        cl_command_queue queue = mkldnn_stream->get_ocl_command_queue();
        cl_program program = clCreateProgramWithSource(
            eng.get_ocl_context(), 1, &source_code, &source_code_length, &errcode);
        if (errcode != CL_SUCCESS)
        {
            throw ngraph_error("Build OpenCL program error: " + to_string(errcode));
        }

        errcode = clBuildProgram(program, 1, &device, "", NULL, NULL);
        if (errcode != CL_SUCCESS)
        {
            size_t log_length = 0;
            int info_errcode =
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, 0, &log_length);
            if (info_errcode != CL_SUCCESS)
            {
                throw ngraph_error("clGetProgramBuildInfo(log_length) error: " +
                                   to_string(info_errcode));
            }

            void* log = ngraph_malloc(log_length);
            info_errcode =
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_length, log, 0);
            if (info_errcode != CL_SUCCESS)
            {
                throw ngraph_error("clGetProgramBuildInfo(log) error: " + to_string(info_errcode));
            }

            string err_string((const char*)log);
            ngraph_free(log);

            throw ngraph_error("Error during the build of OpenCL program. Error: " +
                               to_string(errcode) + "\nBuild log:" + err_string);
        }

        cl_kernel kernel = clCreateKernel(program, kr.m_entry_point.c_str(), &errcode);
        if (errcode != CL_SUCCESS)
        {
            throw ngraph_error("Create OpenCL kernel error: " + to_string(errcode));
        }

        //kr.kernel = kernel;
#else
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
#endif
        ++m_count_krnls;
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
