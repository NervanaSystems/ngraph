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

#include <iomanip>
#include <sys/resource.h>
#include <sys/time.h>

#include <CPP/activation.hpp>
#include <CPP/activation_grad.hpp>
#include <CPP/arg_max_min.hpp>
#include <CPP/batch_norm.hpp>
#include <CPP/border.hpp>
#include <CPP/broadcast.hpp>
#include <CPP/concatenation.hpp>
#include <CPP/convolution.hpp>
#include <CPP/convolution_grad_input.hpp>
#include <CPP/crop.hpp>
#include <CPP/data.hpp>
#include <CPP/eltwise.hpp>
#include <CPP/gemm.hpp>
#include <CPP/input_layout.hpp>
#include <CPP/layout.hpp>
#include <CPP/lrn.hpp>
#include <CPP/mutable_data.hpp>
#include <CPP/permute.hpp>
#include <CPP/pooling.hpp>
#include <CPP/reorder.hpp>
#include <CPP/reshape.hpp>
#include <CPP/scale.hpp>
#include <CPP/select.hpp>
#include <CPP/softmax.hpp>
#include <CPP/topology.hpp>

#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/any_all_replacement.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_batchnorm.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_broadcast.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_convolution.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_func_call.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_softmax.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"
#include "ngraph/runtime/intelgpu/visualize_tree.hpp"

#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/parameter_vector.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using intelgpu_space = runtime::intelgpu::IntelGPULayout;

#define USE_INTELGPU_CUSTOM_KERNELS 0

// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// Abs,
// Acos,
// ...
#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/op_tbl.hpp"
};
#undef NGRAPH_OP

static OP_TYPEID get_typeid(const string& s)
{
// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// {"Abs", OP_TYPEID::Abs},
// {"Acos", OP_TYPEID::Acos},
// ...
#define NGRAPH_OP(a, b) {#a, OP_TYPEID::a},
    static const unordered_map<string, OP_TYPEID> typeid_map{
#include "ngraph/op/op_tbl.hpp"
    };
#undef NGRAPH_OP
    auto it = typeid_map.find(s);
    if (it == typeid_map.end())
    {
        throw unsupported_op("Unsupported op '" + s + "'");
    }
    return it->second;
}

static void arguments_check(const shared_ptr<Node>& op, size_t input, size_t output)
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

static void
    memory_size_check(size_t memory_size, const shared_ptr<Node>& node, const string& function_name)
{
    const size_t tensor_size = shape_size(node->get_shape()) * node->get_element_type().size();

    if (memory_size != tensor_size)
    {
        ostringstream os;
        os << "IntelGPU backend failed memory check. In \"" << function_name << "\" with Node \""
           << node->get_name() << "\" and " << node->get_shape() << " mismatched memory sizes "
           << tensor_size << " and " << memory_size;
        throw invalid_argument(os.str());
    }
}

static const string& get_input_name(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_tensor().get_name();
}

static const string& get_output_name(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_outputs().at(num).get_tensor().get_name();
}

static const Shape& get_input_shape(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_shape();
}

static const Shape& get_output_shape(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_outputs().at(num).get_shape();
}

static const element::Type& get_input_type(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_tensor().get_element_type();
}

static const element::Type& get_output_type(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_outputs().at(num).get_tensor().get_element_type();
}

static void do_eltwise_operation(cldnn::topology& topology,
                                 const shared_ptr<Node>& op,
                                 cldnn::eltwise_mode mode)
{
    arguments_check(op, 2, 1);

    if ((get_input_type(op) == element::i32 || get_input_type(op) == element::i64) &&
        (mode == cldnn::eltwise_mode::min || mode == cldnn::eltwise_mode::max))
    {
        string custom_op;

        if (mode == cldnn::eltwise_mode::min)
        {
            custom_op = "min";
        }
        else if (mode == cldnn::eltwise_mode::max)
        {
            custom_op = "max";
        }
        else
        {
            custom_op = "not_implemented_operation";
        }

        runtime::intelgpu::do_eltwise_kernel(topology,
                                             get_input_name(op, 0),
                                             get_input_shape(op, 0),
                                             get_input_type(op, 0),
                                             get_input_name(op, 1),
                                             get_input_shape(op, 1),
                                             get_output_name(op),
                                             get_output_shape(op),
                                             get_output_type(op),
                                             custom_op);
    }
    else
    {
        const cldnn::eltwise op_add(
            get_output_name(op), {get_input_name(op, 0), get_input_name(op, 1)}, mode);
        topology.add(op_add);
    }
}

static void do_unary_operation(cldnn::topology& topology,
                               const shared_ptr<Node>& op,
                               cldnn_activation_func mode,
                               const cldnn_activation_additional_params& param = {0.f, 0.f})
{
    arguments_check(op, 1, 1);

    const cldnn::activation cldnn_unary(get_output_name(op), get_input_name(op), mode, param);
    topology.add(cldnn_unary);
}

static void do_pooling_operation(cldnn::topology& topology,
                                 const shared_ptr<Node>& op,
                                 const Shape& pool_shape,
                                 const Strides& pool_strides,
                                 const Shape& pad_below,
                                 const cldnn::pooling_mode mode)
{
    arguments_check(op, 1, 1);

    const cldnn::tensor output_size = intelgpu_space::create_cldnn_tensor(get_output_shape(op));
    const cldnn::tensor input_offset = intelgpu_space::create_cldnn_offset(pad_below);
    const cldnn::tensor size = intelgpu_space::create_cldnn_tensor(pool_shape);
    const cldnn::tensor stride = intelgpu_space::create_cldnn_tensor(pool_strides);

    const cldnn::pooling cldnn_pooling(
        get_output_name(op), get_input_name(op), mode, size, stride, input_offset, output_size);
    topology.add(cldnn_pooling);
}

static void do_logical_operation(cldnn::topology& topology,
                                 const shared_ptr<Node>& op,
                                 const string& operation)
{
    arguments_check(op, 2, 1);

    runtime::intelgpu::do_logic_kernel(topology,
                                       get_input_name(op, 0),
                                       get_input_shape(op, 0),
                                       get_input_type(op, 0),
                                       get_input_name(op, 1),
                                       get_input_shape(op, 1),
                                       get_output_name(op),
                                       get_output_shape(op),
                                       get_output_type(op),
                                       operation);
}

// This function needed to only change the name of the data in topology
// No real data copy needed
static void do_equal_propagation(cldnn::topology& topology,
                                 const string& input_name,
                                 const string& output_name)
{
    const vector<cldnn::primitive_id> input_names(1, input_name);

    const cldnn::concatenation op_concat(output_name, input_names, cldnn::concatenation::along_x);
    topology.add(op_concat);
}

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::intelgpu::IntelGPUBackend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

static size_t get_max_memory_rss()
{
    size_t result = 0;
    struct rusage usage;

    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        result = usage.ru_maxrss; // the value is in kilobytes

        // aligne result to return bytes
        result *= 1000;
    }

    return result;
}

runtime::intelgpu::IntelGPUBackend::IntelGPUBackend()
{
    bool profiling = false;

    // This should be used to allow nbench work with "--timing_detail" option
    if (getenv("NGRAPH_INTELGPU_STAT") != nullptr)
    {
        profiling = true;
    }

    // Print out default profile and statistic to the output
    if (getenv("NGRAPH_INTELGPU_PROFILE") != nullptr)
    {
        profiling = true;
    }

    // Control the number of lines in ::call profile
    const char* profile_lines_count = getenv("NGRAPH_INTELGPU_PROFILE_LINES");
    if (profile_lines_count != nullptr)
    {
        profiling = true;
    }

    cldnn::engine_configuration cldnn_configuration(profiling);
    ocl_engine = make_shared<cldnn::engine>(cldnn_configuration);
}

shared_ptr<runtime::Tensor>
    runtime::intelgpu::IntelGPUBackend::create_tensor(const element::Type& element_type,
                                                      const Shape& shape)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(
        element_type, shape, *ocl_engine, nullptr, this);
}

shared_ptr<runtime::Tensor> runtime::intelgpu::IntelGPUBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(
        element_type, shape, *ocl_engine, memory_pointer, this);
}

unique_ptr<runtime::Executable>
    runtime::intelgpu::IntelGPUBackend::compile(shared_ptr<Function> func,
                                                bool enable_performance_collection)
{
    std::unique_ptr<IntelGPUExecutable> exec{
        new IntelGPUExecutable(function, enable_performance_collection)};

    return exec;
}

// The cldnn::network contains something like "generic_layer_0_Parameter_254_0" names
// This function should return "Parameter_254" from the example above
static string convert_cldnn_names(shared_ptr<Function> func, const string& cldnn_name)
{
    const string key("_");
    string result;

    const size_t last_key = cldnn_name.rfind(key);
    const size_t pre_last_key = cldnn_name.rfind(key, last_key - 1);
    const size_t pre_pre_last_key = cldnn_name.rfind(key, pre_last_key - 1);

    if (pre_pre_last_key == std::string::npos)
    {
        result = cldnn_name.substr(0, last_key);
    }
    else
    {
        result = cldnn_name.substr(pre_pre_last_key + 1, last_key - pre_pre_last_key - 1);
    }

    return result;
}

static Node* get_node_by_name(const shared_ptr<Function> func, const string& name)
{
    for (shared_ptr<Node> node : func->get_ops())
    {
        if (node->get_name() == name)
        {
            return node.get();
        }
    }

    return nullptr;
}
