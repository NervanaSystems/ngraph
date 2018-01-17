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

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/acos.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/asin.hpp"
#include "ngraph/ops/atan.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/ceiling.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/cos.hpp"
#include "ngraph/ops/cosh.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/ops/floor.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/minimum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/sign.hpp"
#include "ngraph/ops/sin.hpp"
#include "ngraph/ops/sinh.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sqrt.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/tan.hpp"
#include "ngraph/ops/tanh.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"

using namespace std;
using namespace ngraph;

static const string s_output_dir = "gpu_codegen";

class StaticInitializers
{
public:
    StaticInitializers() { ngraph::file_util::remove_directory(s_output_dir); }
};

static string emit_string_array(const vector<string>& s, size_t max_line_length)
{
    stringstream ss;
    stringstream line;
    for (size_t i = 0; i < s.size(); i++)
    {
        if (i != 0)
        {
            line << ",";
        }
        stringstream value;
        value << s[i];
        string value_string = value.str();
        if (static_cast<size_t>(line.tellp()) + value_string.size() + 1 <= max_line_length)
        {
            if (i > 0)
            {
                line << " ";
            }
            line << value_string;
        }
        else
        {
            ss << line.str() << "\n";
            line.str("");
            line << value_string;
        }
    }
    ss << line.str();
    return ss.str();
}

static StaticInitializers s_static_initializers;

#define TI(x) type_index(typeid(x))

static const runtime::gpu::OpMap dispatcher{
    {TI(ngraph::op::Add), &runtime::gpu::GPU_Emitter::EmitAdd},
    {TI(ngraph::op::Dot), &runtime::gpu::GPU_Emitter::EmitDot},
    {TI(ngraph::op::Multiply), &runtime::gpu::GPU_Emitter::EmitMultiply},
    {TI(ngraph::op::Parameter), &runtime::gpu::GPU_Emitter::EmitNop},
    {TI(ngraph::op::Abs), &runtime::gpu::GPU_Emitter::EmitAbs},
    {TI(ngraph::op::Concat), &runtime::gpu::GPU_Emitter::EmitConcat},
    {TI(ngraph::op::Divide), &runtime::gpu::GPU_Emitter::EmitDivide},
    {TI(ngraph::op::Equal), &runtime::gpu::GPU_Emitter::EmitEqual},
    {TI(ngraph::op::Greater), &runtime::gpu::GPU_Emitter::EmitGreater},
    {TI(ngraph::op::GreaterEq), &runtime::gpu::GPU_Emitter::EmitGreaterEq},
    {TI(ngraph::op::Less), &runtime::gpu::GPU_Emitter::EmitLess},
    {TI(ngraph::op::LessEq), &runtime::gpu::GPU_Emitter::EmitLessEq},
    {TI(ngraph::op::Log), &runtime::gpu::GPU_Emitter::EmitLog},
    {TI(ngraph::op::Maximum), &runtime::gpu::GPU_Emitter::EmitMaximum},
    {TI(ngraph::op::Minimum), &runtime::gpu::GPU_Emitter::EmitMinimum},
    {TI(ngraph::op::Negative), &runtime::gpu::GPU_Emitter::EmitNegative},
    {TI(ngraph::op::NotEqual), &runtime::gpu::GPU_Emitter::EmitNotEqual},
    {TI(ngraph::op::Power), &runtime::gpu::GPU_Emitter::EmitPower},
    {TI(ngraph::op::Select), &runtime::gpu::GPU_Emitter::EmitSelect},
    {TI(ngraph::op::Subtract), &runtime::gpu::GPU_Emitter::EmitSubtract},
    {TI(ngraph::op::Broadcast), &runtime::gpu::GPU_Emitter::EmitBroadcast},
    {TI(ngraph::op::Convert), &runtime::gpu::GPU_Emitter::EmitConvert},
    {TI(ngraph::op::Constant), &runtime::gpu::GPU_Emitter::EmitConstant},
    {TI(ngraph::op::Reshape), &runtime::gpu::GPU_Emitter::EmitReshape},
    {TI(ngraph::op::FunctionCall), &runtime::gpu::GPU_Emitter::EmitFunctionCall},
    {TI(ngraph::op::Reduce), &runtime::gpu::GPU_Emitter::EmitReduce},
    {TI(ngraph::op::Sign), &runtime::gpu::GPU_Emitter::EmitSign},
    {TI(ngraph::op::Slice), &runtime::gpu::GPU_Emitter::EmitSlice},
    {TI(ngraph::op::Sum), &runtime::gpu::GPU_Emitter::EmitSum},
    {TI(ngraph::op::Exp), &runtime::gpu::GPU_Emitter::EmitExp},
    {TI(ngraph::op::Sin), &runtime::gpu::GPU_Emitter::EmitSin},
    {TI(ngraph::op::Sinh), &runtime::gpu::GPU_Emitter::EmitSinh},
    {TI(ngraph::op::Cos), &runtime::gpu::GPU_Emitter::EmitCos},
    {TI(ngraph::op::Cosh), &runtime::gpu::GPU_Emitter::EmitCosh},
    {TI(ngraph::op::Tan), &runtime::gpu::GPU_Emitter::EmitTan},
    {TI(ngraph::op::Tanh), &runtime::gpu::GPU_Emitter::EmitTanh},
    {TI(ngraph::op::Asin), &runtime::gpu::GPU_Emitter::EmitAsin},
    {TI(ngraph::op::Acos), &runtime::gpu::GPU_Emitter::EmitAcos},
    {TI(ngraph::op::Atan), &runtime::gpu::GPU_Emitter::EmitAtan},
    {TI(ngraph::op::ReplaceSlice), &runtime::gpu::GPU_Emitter::EmitReplaceSlice},
    {TI(ngraph::op::OneHot), &runtime::gpu::GPU_Emitter::EmitOneHot},
    {TI(ngraph::op::Floor), &runtime::gpu::GPU_Emitter::EmitFloor},
    {TI(ngraph::op::Ceiling), &runtime::gpu::GPU_Emitter::EmitCeiling},
    {TI(ngraph::op::Sqrt), &runtime::gpu::GPU_Emitter::EmitSqrt},
    {TI(ngraph::op::Convolution), &runtime::gpu::GPU_Emitter::EmitConvolution},
    {TI(ngraph::op::Not), &runtime::gpu::GPU_Emitter::EmitNot},
    {TI(ngraph::op::MaxPool), &runtime::gpu::GPU_Emitter::EmitMaxPool},
    {TI(ngraph::op::Reverse), &runtime::gpu::GPU_Emitter::EmitReverse},
};

runtime::gpu::GPU_ExternalFunction::GPU_ExternalFunction(
    const shared_ptr<ngraph::Function>& function, bool release_function)
    : ngraph::runtime::ExternalFunction(function, release_function)
    , m_compiled_function(nullptr)
    , m_emit_timing(std::getenv("NGRAPH_GPU_EMIT_TIMING") != nullptr)
    , m_use_tbb(std::getenv("NGRAPH_GPU_USE_TBB") != nullptr)
{
}

void runtime::gpu::GPU_ExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }

    string function_name = m_function->get_name();
    string dump_filename = file_util::path_join(s_output_dir, function_name + "_ops.txt");

    pass::Manager pass_manager;
    // pass_manager.register_pass<pass::TopologicalSort>();
    // For now, just make everyone row-major.
    pass_manager.register_pass<pass::AssignLayout<descriptor::layout::DenseTensorViewLayout>>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.register_pass<pass::MemoryLayout>(64);
    pass_manager.register_pass<pass::DumpSorted>(dump_filename);
    pass_manager.run_passes(m_function);

    GPU_Emitter emitter;
    codegen::CodeWriter& writer = emitter.get_code_writer();

    writer +=
        R"(// Generated by the NGraph GPU backend
    #include <cassert>
    #include <cmath>
    #include <cstdlib>
    #include <fstream>
    #include <fstream>
    #include <iostream>
    #include <memory>
    #include <string>
    #include <tuple>
    #include <typeindex>
    #include <typeinfo>
    #include <unordered_map>

    #include "cuda.h"
    #include "ngraph/codegen/code_writer.hpp"
    #include "ngraph/codegen/compiler.hpp"
    #include "ngraph/codegen/execution_engine.hpp"
    #include "ngraph/descriptor/input.hpp"
    #include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
    #include "ngraph/descriptor/output.hpp"
    #include "ngraph/descriptor/primary_tensor_view.hpp"
    #include "ngraph/file_util.hpp"
    #include "ngraph/function.hpp"
    #include "ngraph/graph_util.hpp"
    #include "ngraph/node.hpp"
    #include "ngraph/ops/abs.hpp"
    #include "ngraph/ops/acos.hpp"
    #include "ngraph/ops/add.hpp"
    #include "ngraph/ops/asin.hpp"
    #include "ngraph/ops/atan.hpp"
    #include "ngraph/ops/broadcast.hpp"
    #include "ngraph/ops/ceiling.hpp"
    #include "ngraph/ops/concatenate.hpp"
    #include "ngraph/ops/constant.hpp"
    #include "ngraph/ops/convert.hpp"
    #include "ngraph/ops/convolution.hpp"
    #include "ngraph/ops/cos.hpp"
    #include "ngraph/ops/cosh.hpp"
    #include "ngraph/ops/divide.hpp"
    #include "ngraph/ops/dot.hpp"
    #include "ngraph/ops/equal.hpp"
    #include "ngraph/ops/exp.hpp"
    #include "ngraph/ops/floor.hpp"
    #include "ngraph/ops/function_call.hpp"
    #include "ngraph/ops/greater.hpp"
    #include "ngraph/ops/greater_eq.hpp"
    #include "ngraph/ops/less.hpp"
    #include "ngraph/ops/less_eq.hpp"
    #include "ngraph/ops/log.hpp"
    #include "ngraph/ops/max_pool.hpp"
    #include "ngraph/ops/maximum.hpp"
    #include "ngraph/ops/minimum.hpp"
    #include "ngraph/ops/multiply.hpp"
    #include "ngraph/ops/negative.hpp"
    #include "ngraph/ops/not.hpp"
    #include "ngraph/ops/not_equal.hpp"
    #include "ngraph/ops/one_hot.hpp"
    #include "ngraph/ops/power.hpp"
    #include "ngraph/ops/reduce.hpp"
    #include "ngraph/ops/replace_slice.hpp"
    #include "ngraph/ops/reshape.hpp"
    #include "ngraph/ops/reverse.hpp"
    #include "ngraph/ops/select.hpp"
    #include "ngraph/ops/sign.hpp"
    #include "ngraph/ops/sin.hpp"
    #include "ngraph/ops/sinh.hpp"
    #include "ngraph/ops/slice.hpp"
    #include "ngraph/ops/sqrt.hpp"
    #include "ngraph/ops/subtract.hpp"
    #include "ngraph/ops/sum.hpp"
    #include "ngraph/ops/tan.hpp"
    #include "ngraph/ops/tanh.hpp"
    #include "ngraph/pass/assign_layout.hpp"
    #include "ngraph/pass/dump_sorted.hpp"
    #include "ngraph/pass/liveness.hpp"
    #include "ngraph/pass/manager.hpp"
    #include "ngraph/pass/memory_layout.hpp"
    #include "ngraph/runtime/aligned_buffer.hpp"
    #include "ngraph/util.hpp"
)";

    string pch_header_source = writer.get_code();

    writer += R"(
    using namespace ngraph::runtime;
    using namespace std;

    void check_cuda_errors(CUresult err) {
      assert(err == CUDA_SUCCESS);
      // assert(err == err);
    }


)";

    //     // The "dso_handle" symbol is required by __cxa_atexit()
    //     // which is enabled because the JIT uses it as the default mechanism
    //     // to register cleanup handlers. We use it, and not atexit(), because
    //     // atexit() happens too late, when the JIT is no longer alive

    writer << "void *__dso_handle = 0;\n\n";
    writer << "// Declare all constants\n";
    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            const op::Constant* c = dynamic_cast<op::Constant*>(node.get());
            if (c)
            {
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
                auto c_value_strings = c->get_value_strings();
                writer << "static " << tv->get_tensor().get_element_type().c_type_string() << " "
                       << tv->get_tensor().get_name() << "[" << c_value_strings.size() << "] =\n";
                writer << "{\n";
                writer.indent++;
                writer << emit_string_array(c_value_strings, 100 - writer.indent * 4);
                writer.indent--;
                writer << "\n};\n\n";
                m_variable_name_map[tv->get_tensor().get_name()] = tv->get_tensor().get_name();
            }
        }
    }

    writer << "// Declare all functions\n";
    for (shared_ptr<Function> f : pass_manager.get_state().get_functions())
    {
        writer << "extern \"C\" void " << f->get_name() << "(void** inputs, void** outputs);\n";
    }
    writer << "\n";
    writer << "extern \"C\" void " << pass_manager.get_state().get_functions()[0]->get_name()
           << "(void** inputs, void** outputs){\n";
    writer += R"(
    CUdevice    device;
    CUmodule    cuda_module;
    CUcontext   context;
    CUfunction  add_function;
    CUfunction  mult_function;
    CUlinkState linker;
    int         dev_count;
    check_cuda_errors(cuInit(0));
    check_cuda_errors(cuDeviceGetCount(&dev_count));
    check_cuda_errors(cuDeviceGet(&device, 0));

    // char name[128];
    // check_cuda_errors(cuDeviceGetName(name, 128, device));
    // std::cout << "Using CUDA Device [0]: " << name << "\n";

    // int dev_major, dev_minor;
    // check_cuda_errors(cuDeviceComputeCapability(&dev_major, &dev_minor, device));
    // std::cout << "Device Compute Capability: "
    //           << dev_major << "." << dev_minor << "\n";
    // if (dev_major < 2) {
    //   std::cerr << "ERROR: Device 0 is not SM 2.0 or greater\n";
    // }

    const auto kernels = R"#(
    .version 5.0
    .target sm_60
    .address_size 64

      // .globl	_Z7ew_multPfS_S_ // -- Begin function _Z7ew_multPfS_S_
    .global .align 1 .b8 threadIdx[1];
                                            // @_Z7ew_multPfS_S_
    .visible .entry _Z7ew_multPfS_S_(
      .param .u64 _Z7ew_multPfS_S__param_0,
      .param .u64 _Z7ew_multPfS_S__param_1,
      .param .u64 _Z7ew_multPfS_S__param_2
    )
    {
      .local .align 8 .b8 	__local_depot0[24];
      .reg .b64 	%SP;
      .reg .b64 	%SPL;
      .reg .f32 	%f<4>;
      .reg .b32 	%r<2>;
      .reg .b64 	%rd<17>;

    // BB#0:
      mov.u64 	%SPL, __local_depot0;
      cvta.local.u64 	%SP, %SPL;
      ld.param.u64 	%rd3, [_Z7ew_multPfS_S__param_2];
      ld.param.u64 	%rd2, [_Z7ew_multPfS_S__param_1];
      ld.param.u64 	%rd1, [_Z7ew_multPfS_S__param_0];
      cvta.to.global.u64 	%rd4, %rd3;
      cvta.global.u64 	%rd5, %rd4;
      cvta.to.global.u64 	%rd6, %rd2;
      cvta.global.u64 	%rd7, %rd6;
      cvta.to.global.u64 	%rd8, %rd1;
      cvta.global.u64 	%rd9, %rd8;
      st.u64 	[%SP+0], %rd9;
      st.u64 	[%SP+8], %rd7;
      st.u64 	[%SP+16], %rd5;
      ld.u64 	%rd10, [%SP+0];
      mov.u32 	%r1, %tid.x;
      mul.wide.u32 	%rd11, %r1, 4;
      add.s64 	%rd12, %rd10, %rd11;
      ld.f32 	%f1, [%rd12];
      ld.u64 	%rd13, [%SP+8];
      add.s64 	%rd14, %rd13, %rd11;
      ld.f32 	%f2, [%rd14];
      mul.rn.f32 	%f3, %f1, %f2;
      ld.u64 	%rd15, [%SP+16];
      add.s64 	%rd16, %rd15, %rd11;
      st.f32 	[%rd16], %f3;
      ret;
    }
                                            // -- End function
      // .globl	_Z6ew_addPfS_S_ // -- Begin function _Z6ew_addPfS_S_
    .visible .entry _Z6ew_addPfS_S_(
      .param .u64 _Z6ew_addPfS_S__param_0,
      .param .u64 _Z6ew_addPfS_S__param_1,
      .param .u64 _Z6ew_addPfS_S__param_2
    )                                       // @_Z6ew_addPfS_S_
    {
      .local .align 8 .b8 	__local_depot1[24];
      .reg .b64 	%SP;
      .reg .b64 	%SPL;
      .reg .f32 	%f<4>;
      .reg .b32 	%r<2>;
      .reg .b64 	%rd<17>;

    // BB#0:
      mov.u64 	%SPL, __local_depot1;
      cvta.local.u64 	%SP, %SPL;
      ld.param.u64 	%rd3, [_Z6ew_addPfS_S__param_2];
      ld.param.u64 	%rd2, [_Z6ew_addPfS_S__param_1];
      ld.param.u64 	%rd1, [_Z6ew_addPfS_S__param_0];
      cvta.to.global.u64 	%rd4, %rd3;
      cvta.global.u64 	%rd5, %rd4;
      cvta.to.global.u64 	%rd6, %rd2;
      cvta.global.u64 	%rd7, %rd6;
      cvta.to.global.u64 	%rd8, %rd1;
      cvta.global.u64 	%rd9, %rd8;
      st.u64 	[%SP+0], %rd9;
      st.u64 	[%SP+8], %rd7;
      st.u64 	[%SP+16], %rd5;
      ld.u64 	%rd10, [%SP+0];
      mov.u32 	%r1, %tid.x;
      mul.wide.u32 	%rd11, %r1, 4;
      add.s64 	%rd12, %rd10, %rd11;
      ld.f32 	%f1, [%rd12];
      ld.u64 	%rd13, [%SP+8];
      add.s64 	%rd14, %rd13, %rd11;
      ld.f32 	%f2, [%rd14];
      add.rn.f32 	%f3, %f1, %f2;
      ld.u64 	%rd15, [%SP+16];
      add.s64 	%rd16, %rd15, %rd11;
      st.f32 	[%rd16], %f3;
      ret;
    }
                                            // -- End function
    )#";
    // Create driver context
    check_cuda_errors(cuCtxCreate(&context, 0, device));

    // Create module for object
    check_cuda_errors(cuModuleLoadDataEx(&cuda_module, kernels, 0, 0, 0));

    // Get kernel function
    check_cuda_errors(cuModuleGetFunction(&add_function, cuda_module, "_Z6ew_addPfS_S_"));
    check_cuda_errors(cuModuleGetFunction(&mult_function, cuda_module, "_Z7ew_multPfS_S_"));

    // Device data
    CUdeviceptr dev_bufferA;
    CUdeviceptr dev_bufferB;
    CUdeviceptr dev_bufferC;

    check_cuda_errors(cuMemAlloc(&dev_bufferA, sizeof(float) * 4));
    check_cuda_errors(cuMemAlloc(&dev_bufferB, sizeof(float) * 4));
    check_cuda_errors(cuMemAlloc(&dev_bufferC, sizeof(float) * 4));

    float* host_A = new float[4];
    float* host_B = new float[4];
    float* host_C = new float[4];

    // Populate input
    memcpy(host_A, (float*)(inputs[0]), sizeof(float) * 4);
    memcpy(host_B, (float*)(inputs[1]), sizeof(float) * 4);
    memcpy(host_C, (float*)(inputs[2]), sizeof(float) * 4);

    check_cuda_errors(cuMemcpyHtoD(dev_bufferA, &host_A[0], sizeof(float) * 4));
    check_cuda_errors(cuMemcpyHtoD(dev_bufferB, &host_B[0], sizeof(float) * 4));
    // check_cuda_errors(cuMemcpyHtoD(dev_bufferC, &host_C[0], sizeof(float) * 4));

    unsigned block_size_X = 4;
    unsigned block_size_Y = 1;
    unsigned block_size_Z = 1;
    unsigned grid_size_X = 1;
    unsigned grid_size_Y = 1;
    unsigned grid_size_Z = 1;

    // Kernel parameters
    void* kernel_params[] = {&dev_bufferA, &dev_bufferB, &dev_bufferC};

    // Add Kernel launch
    check_cuda_errors(cuLaunchKernel(add_function,
                                        grid_size_X,
                                        grid_size_Y,
                                        grid_size_Z,
                                        block_size_X,
                                        block_size_Y,
                                        block_size_Z,
                                        0,
                                        NULL,
                                        kernel_params,
                                        NULL));

    check_cuda_errors(cuMemcpyDtoH(&host_A[0], dev_bufferC, sizeof(float) * 4));
    host_B = &host_C[0];
      check_cuda_errors(cuMemcpyHtoD(dev_bufferA, &host_A[0], sizeof(float) * 4));
      check_cuda_errors(cuMemcpyHtoD(dev_bufferB, &host_B[0], sizeof(float) * 4));

    // Mult Kernel launch
    check_cuda_errors(cuLaunchKernel(mult_function,
                                        grid_size_X,
                                        grid_size_Y,
                                        grid_size_Z,
                                        block_size_X,
                                        block_size_Y,
                                        block_size_Z,
                                        0,
                                        NULL,
                                        kernel_params,
                                        NULL));

    // Write final output 
    check_cuda_errors(cuMemcpyDtoH(&((float*)(outputs[0]))[0], dev_bufferC, sizeof(float) * 4));
    // Clean up after ourselves

    // // Clean-up must do this in tensor view!!!

    check_cuda_errors(cuMemFree(dev_bufferA));
    check_cuda_errors(cuMemFree(dev_bufferB));
    check_cuda_errors(cuMemFree(dev_bufferC));
    check_cuda_errors(cuModuleUnload(cuda_module));
    check_cuda_errors(cuCtxDestroy(context));)";

        if (m_emit_timing)
        {
            writer << "// Declare debug timers\n";
            vector<string> names;
            for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
            {
                for (shared_ptr<Node> node : current_function->get_ordered_ops())
                {
                    if (!node->is_parameter() && !node->is_constant())
                    {
                        names.push_back(node->get_name());
                    }
                }
            }
            for (const string& s : names)
            {
                writer << "ngraph::stopwatch timer_" << s << ";\n";
            }
            writer << "extern \"C\" size_t get_debug_timer_count() { return " << names.size()
                   << "; }\n";
            writer << "extern \"C\" const char* get_debug_timer_name(size_t index)\n";
            writer << "{\n";
            writer.indent++;
            writer << "const char* rc;\n";
            writer << "switch(index)\n";
            writer << "{\n";
            for (size_t i = 0; i < names.size(); i++)
            {
                writer << "case " << i << ": rc = \"" << names[i] << "\"; break;\n";
            }
            writer << "default: rc = \"\";\n";
            writer << "}\n";
            writer << "return rc;\n";
            writer.indent--;
            writer << "}\n";
            writer << "extern \"C\" const size_t get_debug_timer_microseconds(size_t index)\n";
            writer << "{\n";
            writer.indent++;
            writer << "size_t rc;\n";
            writer << "switch(index)\n";
            writer << "{\n";
            for (size_t i = 0; i < names.size(); i++)
            {
                writer << "case " << i << ": rc = timer_" << names[i]
                       << ".get_total_microseconds(); break;\n";
            }
            writer << "default: rc = 0;\n";
            writer << "}\n";
            writer << "return rc;\n";
            writer.indent--;
            writer << "}\n";
            writer << "extern \"C\" const size_t get_debug_timer_call_count(size_t index)\n";
            writer << "{\n";
            writer.indent++;
            writer << "size_t rc;\n";
            writer << "switch(index)\n";
            writer << "{\n";
            for (size_t i = 0; i < names.size(); i++)
            {
                writer << "case " << i << ": rc = timer_" << names[i]
                       << ".get_call_count(); break;\n";
            }
            writer << "default: rc = 0;\n";
            writer << "}\n";
            writer << "return rc;\n";
            writer.indent--;
            writer << "}\n";
            writer << "\n";
        }

        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            if (node->liveness_new_list.size() > 0)
            {
                temporaries_used = true;
                for (descriptor::Tensor* tensor : node->liveness_new_list)
                {
                    worst_case_tmp_size += tensor->size();
                }
            }
        }

        if (temporaries_used)
        {
            size_t temp_pool_size = current_function->get_temporary_pool_size();
            writer << "// Allocate the memory pool\n";
            writer << "// Memory pool size is " << temp_pool_size << " bytes\n";
            writer << "// Worst case size is " << worst_case_tmp_size << " bytes\n";
            writer << "ngraph::runtime::AlignedBuffer memory_handler(" << temp_pool_size << ", "
                   << ngraph::runtime::gpu::alignment << ");\n";
            writer << "size_t pool_gpu_ptr = (size_t)memory_handler.get_ptr();\n";
            writer << "\n";

            // Add temporaries to the variable name map
            for (shared_ptr<Node> node : current_function->get_ordered_ops())
            {
                for (descriptor::Tensor* tensor : node->liveness_new_list)
                {
                    stringstream ss;
                    ss << "((" << tensor->get_element_type().c_type_string() << "*)(pool_gpu_ptr + "
                       << tensor->get_pool_offset() << "))";
                    m_variable_name_map[tensor->get_name()] = ss.str();
                }
            }
        }

        // Add inputs to the variable name map
        size_t arg_index = 0;
        for (shared_ptr<op::Parameter> param : current_function->get_parameters())
        {
            for (size_t i = 0; i < param->get_output_size(); ++i)
            {
                shared_ptr<descriptor::TensorView> tv = param->get_output_tensor_view(i);
                const element::Type& et = tv->get_tensor_view_type()->get_element_type();
                string type = et.c_type_string();
                stringstream ss;
                ss << "((" << type << "*)(inputs[" << arg_index << "]))";
                m_variable_name_map[tv->get_tensor().get_name()] = ss.str();
                arg_index++;
            }
        }

        // create output alias map
        size_t output_index = 0;
        unordered_map<descriptor::TensorView*, vector<size_t>> output_alias_map;
        vector<size_t> aliases;
        for (size_t i = 0; i < current_function->get_output_size(); ++i)
        {
            shared_ptr<Node> op = current_function->get_output_op(i);
            shared_ptr<descriptor::TensorView> otv = op->get_output_tensor_view();
            vector<size_t>& al = output_alias_map[otv.get()];
            al.push_back(output_index);
            if (al.size() > 1)
            {
                aliases.push_back(output_index);
            }
            output_index++;
        }

        // Add outputs to the variable name map
        output_index = 0;
        for (size_t i = 0; i < current_function->get_output_size(); ++i)
        {
            shared_ptr<Node> op = current_function->get_output_op(i);
            shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
            const element::Type& et = tv->get_tensor_view_type()->get_element_type();
            bool parameter_as_output = false;
            for (shared_ptr<op::Parameter> param : current_function->get_parameters())
            {
                for (const descriptor::Output& pout : param->get_outputs())
                {
                    shared_ptr<descriptor::TensorView> ptv = pout.get_tensor_view();
                    if (tv == ptv)
                    {
                        parameter_as_output = true;
                        writer << "memcpy(static_cast<" << et.c_type_string() << "*>(outputs["
                               << output_index << "]), "
                               << m_variable_name_map[ptv->get_tensor().get_name()] << ", "
                               << ptv->get_tensor().size() << ");\n";
                        break;
                    }
                }
            }
            if (!parameter_as_output && !contains(aliases, output_index))
            {
                if (contains(constants, tv.get()))
                {
                    writer << "memcpy(outputs[" << output_index << "], "
                           << tv->get_tensor().get_name() << ", " << tv->get_tensor().size()
                           << ");\n";
                }
                else
                {
                    string type = et.c_type_string();
                    stringstream ss;
                    ss << "((" << type << "*)(outputs[" << output_index << "]))";
                    m_variable_name_map[tv->get_tensor().get_name()] = ss.str();
                }
            }
            output_index++;
        }

        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            auto& n = *node; // Work around a compiler warning (*node inside typeid may have effects
            // with shared pointers, which is fine here but clang doesn't like it.)
            auto handler = dispatcher.find(type_index(typeid(n)));
            if (handler == dispatcher.end())
            {
                throw ngraph_error("Unhandled op during code generation : " + node->description());
            }
            vector<GPU_TensorViewWrapper> in;
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                in.push_back(
                    GPU_TensorViewWrapper(tv, m_variable_name_map[tv->get_tensor().get_name()]));
            }
            vector<GPU_TensorViewWrapper> out;
            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                out.push_back(
                    GPU_TensorViewWrapper(tv, m_variable_name_map[tv->get_tensor().get_name()]));
            }

            // Emit operation prologue
            if (!node->is_parameter() && !node->is_constant())
            {
                if (m_use_tbb)
                {
                    writer << "tbb::flow::continue_node<tbb::flow::continue_msg> "
                              "flowgraph_node_"
                           << node->get_name()
                           << "(G, [&](const tbb::flow::continue_msg &msg)\n{\n";
                    writer.indent++;
                }
                if (m_emit_timing)
                {
                    emit_debug_function_entry(writer, node.get(), in, out);
                }
            }

            // Emit operation body
            string func_name;
            auto it = match_functions.find(node.get());
            if (it != match_functions.end())
            {
                func_name = it->second;
            }
            if (func_name.empty())
            {
                handler->second(writer, node.get(), in, out);
            }
            else
            {
                vector<string> names;
                for (const GPU_TensorViewWrapper& tv : in)
                {
                    names.push_back(tv.get_name());
                }
                for (const GPU_TensorViewWrapper& tv : out)
                {
                    names.push_back(tv.get_name());
                }
                writer << func_name << "(" << join(names) << ");\n";
            }

            // Emit operation epilogue
            if (!node->is_parameter() && !node->is_constant())
            {
                handle_output_alias(writer, *node, output_alias_map);
                if (m_emit_timing)
                {
                    emit_debug_function_exit(writer, node.get(), in, out);
                }
                if (m_use_tbb)
                {
                    writer.indent--;
                    writer << "});\n";
                }
            }
        }

        if (m_use_tbb)
        {
            writer << "\n";
            // Build the flow graph
            vector<Node*> dependence_graph_heads;

            traverse_nodes(
                current_function, [&writer, &dependence_graph_heads](shared_ptr<Node> n) {
                    if (!n->is_parameter() && !n->is_constant())
                    {
                        bool is_head = true;
                        for (auto arg : n->get_input_ops())
                        {
                            if (!arg->is_parameter() && !arg->is_constant())
                            {
                                is_head = false;
                                writer << "tbb::flow::make_edge(flowgraph_node_" << arg->get_name()
                                       << ", flowgraph_node_" << n->get_name() << ");\n";
                            }
                        }
                        if (is_head)
                        {
                            dependence_graph_heads.emplace_back(n.get());
                        }
                    }
                });

            writer << "\n";

            // Execute the flow graph
            if (!dependence_graph_heads.empty())
            {
                for (Node* n : dependence_graph_heads)
                {
                    writer << "flowgraph_node_" << n->get_name()
                           << ".try_put(tbb::flow::continue_msg());\n";
                }
                writer << "try { G.wait_for_all(); } catch(...) { throw; }\n";
            }
        }

        writer.indent--;
        // End generated function
        writer += "}\n\n";
    }

    // TODO: Cleanup and make this a utility function

    file_util::make_directory(s_output_dir);
    string filename = file_util::path_join(s_output_dir, function_name + "_codegen.cpp");
    ofstream out(filename);
    string code = writer.get_code();
    out << code;
    out.close();

    m_compiler.reset(new codegen::Compiler());
    m_execution_engine.reset(new codegen::ExecutionEngine());

    m_compiler->set_precompiled_header_source(pch_header_source);

    auto codegen_module = m_compiler->compile(code);

    if (codegen_module == nullptr)
    {
        throw runtime_error("function failed to compile");
    }
    m_execution_engine->add_module(codegen_module);
    m_execution_engine->finalize();
    m_compiled_function = m_execution_engine->find_function<EntryPoint_t>(function_name);
    assert(m_compiled_function);

    m_is_compiled = true;
    if (m_release_function)
    {
        release_function();
    }
}

void runtime::gpu::GPU_ExternalFunction::handle_output_alias(
    codegen::CodeWriter& writer,
    const Node& node,
    const unordered_map<descriptor::TensorView*, vector<size_t>>& output_alias_map)
{
    for (const descriptor::Output& output : node.get_outputs())
    {
        shared_ptr<descriptor::TensorView> otv = output.get_tensor_view();
        auto it = output_alias_map.find(otv.get());
        if (it != output_alias_map.end())
        {
            const vector<size_t>& outputs = it->second;
            if (outputs.size() > 1)
            {
                writer << "{    // handle output alias for previous op\n";
                writer.indent++;
                for (size_t i = 1; i < outputs.size(); i++)
                {
                    writer << "memcpy(static_cast<void*>(outputs[" << outputs[i]
                           << "]), static_cast<void*>(outputs[" << outputs[0] << "]), "
                           << otv->get_tensor().size() << ");\n";
                }
                writer.indent--;
                writer << "}\n";
            }
        }
    }
}

shared_ptr<ngraph::runtime::CallFrame> runtime::gpu::GPU_ExternalFunction::make_call_frame()
{
    if (!m_is_compiled)
    {
        compile();
    }

    return make_shared<ngraph::runtime::gpu::GPU_CallFrame>(shared_from_this(),
                                                            m_compiled_function);
}

void runtime::gpu::GPU_ExternalFunction::emit_debug_function_entry(
    codegen::CodeWriter& writer,
    Node* node,
    const std::vector<GPU_TensorViewWrapper>& in,
    const std::vector<GPU_TensorViewWrapper>& out)
{
    writer << "timer_" << node->get_name() << ".start();\n";
}

void runtime::gpu::GPU_ExternalFunction::emit_debug_function_exit(
    codegen::CodeWriter& writer,
    Node* node,
    const std::vector<GPU_TensorViewWrapper>& in,
    const std::vector<GPU_TensorViewWrapper>& out)
{
    writer << "timer_" << node->get_name() << ".stop();\n";
}
