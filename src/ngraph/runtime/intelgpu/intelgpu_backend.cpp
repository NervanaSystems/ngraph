/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <CPP/activation.hpp>
#include <CPP/activation_grad.hpp>
#include <CPP/batch_norm.hpp>
#include <CPP/convolution.hpp>
#include <CPP/data.hpp>
#include <CPP/eltwise.hpp>
#include <CPP/input_layout.hpp>
#include <CPP/layout.hpp>
#include <CPP/permute.hpp>
#include <CPP/pooling.hpp>
#include <CPP/reorder.hpp>
#include <CPP/reshape.hpp>
#include <CPP/scale.hpp>
#include <CPP/topology.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_batchnorm.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_broadcast.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"

#include "ngraph/node.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static void arguments_check(const shared_ptr<Node>& op, size_t input, size_t output)
{
    if (op->get_input_size() != input || op->get_output_size() != output)
    {
        ostringstream os;
        os << "Operation \"" << op->description() << "\" input and output sizes mismatch."
           << " Expected input size=" << op->get_input_size() << ", provided=" << input
           << ". Expected output size=" << op->get_output_size() << ", provided=" << output;
        throw invalid_argument(os.str());
    }
}

static void do_eltwise_operation(cldnn::topology& topology,
                                 const shared_ptr<Node>& op,
                                 cldnn::eltwise_mode mode)
{
    arguments_check(op, 2, 1);

    vector<cldnn::primitive_id> op_add_inputs;
    for (const descriptor::Input& op_input : op->get_inputs())
    {
        const string& element_name = op_input.get_tensor().get_name();
        op_add_inputs.push_back(element_name);
    }

    const string& output_name = op->get_outputs().begin()->get_tensor().get_name();

    const cldnn::eltwise op_add(output_name, op_add_inputs, mode);
    topology.add(op_add);
}

static void do_unary_operation(cldnn::topology& topology,
                               const shared_ptr<Node>& op,
                               cldnn_activation_func mode,
                               const cldnn_activation_additional_params& param = {0.f, 0.f})
{
    arguments_check(op, 1, 1);

    const string& input_name = op->get_inputs().begin()->get_tensor().get_name();
    const string& output_name = op->get_outputs().begin()->get_tensor().get_name();

    const cldnn::activation cldnn_unary(output_name, input_name, mode, param);
    topology.add(cldnn_unary);
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

runtime::intelgpu::IntelGPUBackend::IntelGPUBackend()
{
    ocl_engine = make_shared<cldnn::engine>();
}

shared_ptr<runtime::TensorView>
    runtime::intelgpu::IntelGPUBackend::create_tensor(const element::Type& element_type,
                                                      const Shape& shape)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(element_type, shape, *ocl_engine);
}

shared_ptr<runtime::TensorView> runtime::intelgpu::IntelGPUBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(
        element_type, shape, *ocl_engine, memory_pointer);
}

bool runtime::intelgpu::IntelGPUBackend::compile(shared_ptr<Function> func)
{
    FunctionInstance& instance = ocl_networks[func];
    if (instance.ocl_network != nullptr)
    {
        return true;
    }

    cldnn::topology topology;

    for (shared_ptr<Node> op : func->get_ops())
    {
        if ("Parameter" == op->description())
        {
            arguments_check(op, 0, 1);

            const string& element_name = op->get_output_tensor_view()->get_tensor().get_name();
            const cldnn::layout element_layout =
                IntelGPULayout::create_cldnn_layout(op->get_element_type(), op->get_shape());

            const cldnn::input_layout op_layout(element_name, element_layout);
            topology.add(op_layout);
        }
        else if ("Result" == op->description())
        {
            arguments_check(op, 1, 1);

            const string& input_name = op->get_inputs().begin()->get_tensor().get_name();
            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();

            do_equal_propagation(topology, input_name, output_name);
        }
        else if ("GetOutputElement" == op->description())
        {
            arguments_check(op, 3, 1);

            const shared_ptr<op::GetOutputElement> elem =
                static_pointer_cast<op::GetOutputElement>(op);
            const string& input_name = op->get_inputs().at(elem->get_n()).get_tensor().get_name();
            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();

            do_equal_propagation(topology, input_name, output_name);
        }
        else if ("Add" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::sum);
        }
        else if ("Multiply" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::prod);
        }
        else if ("Divide" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::div);
        }
        else if ("Maximum" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::max);
        }
        else if ("Constant" == op->description())
        {
            arguments_check(op, 0, 1);

            auto input_it = op->get_outputs().cbegin();
            const descriptor::Tensor& output_tensor = input_it->get_tensor();
            const string& output_name = output_tensor.get_name();
            const shared_ptr<op::Constant> constant_inst = static_pointer_cast<op::Constant>(op);
            void* memory_pointer = const_cast<void*>(constant_inst->get_data_ptr());

            const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(
                output_tensor.get_element_type(), input_it->get_shape());
            const cldnn::memory mem(
                cldnn::memory::attach<void>(layout, memory_pointer, layout.bytes_count()));

            const cldnn::data op_const(output_name, mem);
            topology.add(op_const);
        }
        else if ("Dot" == op->description())
        {
            arguments_check(op, 2, 1);

            const string& inputA_name = op->get_inputs().at(0).get_tensor().get_name();
            const Shape& inputA_shape = op->get_inputs().at(0).get_shape();
            const string& inputB_name = op->get_inputs().at(1).get_tensor().get_name();
            const Shape& inputB_shape = op->get_inputs().at(1).get_shape();
            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();
            const Shape& output_shape = op->get_outputs().begin()->get_shape();
            const element::Type& output_type =
                op->get_outputs().begin()->get_tensor().get_element_type();

            do_dot_operation(topology,
                             inputA_name,
                             inputA_shape,
                             inputB_name,
                             inputB_shape,
                             output_name,
                             output_shape,
                             output_type);
        }
        else if ("MaxPool" == op->description())
        {
            arguments_check(op, 1, 1);

            const string& input_name = op->get_inputs().begin()->get_tensor().get_name();
            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();
            const Shape& out_shape = op->get_outputs().begin()->get_shape();
            const cldnn::tensor output_size =
                runtime::intelgpu::IntelGPULayout::create_cldnn_tensor(out_shape);

            const shared_ptr<op::MaxPool> max_pool = static_pointer_cast<op::MaxPool>(op);
            const Shape& pool_shape = max_pool->get_window_shape();
            const Strides& pool_strides = max_pool->get_window_movement_strides();
            const Shape& pad = max_pool->get_padding_below();

            vector<cldnn::tensor::value_type> offset({0, 0, 0, 0}); // No action by default
            size_t ridx = 4;
            for (auto i = pad.crbegin(); i != pad.crend() && ridx > 0; ++i, --ridx)
            {
                offset.at(ridx - 1) = -(*i);
            }

            const cldnn::tensor input_offset(
                offset.at(0), offset.at(1), offset.at(3), offset.at(2));
            const cldnn::tensor size =
                runtime::intelgpu::IntelGPULayout::create_cldnn_tensor(pool_shape);
            const cldnn::tensor strides =
                runtime::intelgpu::IntelGPULayout::create_cldnn_tensor(pool_strides);

            const cldnn::pooling cldd_pooling(output_name,
                                              input_name,
                                              cldnn::pooling_mode::max,
                                              size,
                                              strides,
                                              input_offset,
                                              output_size);
            topology.add(cldd_pooling);
        }
        else if ("Broadcast" == op->description())
        {
            arguments_check(op, 1, 1);

            const string& input_name = op->get_inputs().begin()->get_tensor().get_name();
            const Shape& input_shape = op->get_inputs().begin()->get_shape();

            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();
            const Shape& output_shape = op->get_outputs().begin()->get_shape();
            const element::Type& output_type =
                op->get_outputs().begin()->get_tensor().get_element_type();

            const shared_ptr<op::Broadcast> broadcast = static_pointer_cast<op::Broadcast>(op);
            const AxisSet& axis = broadcast->get_broadcast_axes();

            if (axis.empty())
            {
                do_equal_propagation(topology, input_name, output_name);
            }
            else if (input_shape.empty())
            {
                do_bcast_sum_operation_scalar(topology,
                                              input_name,
                                              input_shape,
                                              output_name,
                                              output_shape,
                                              output_type,
                                              true);
            }
            else
            {
                do_bcast_sum_operation(topology,
                                       input_name,
                                       input_shape,
                                       output_name,
                                       output_shape,
                                       output_type,
                                       axis,
                                       true);
            }
        }
        else if ("Sum" == op->description())
        {
            arguments_check(op, 1, 1);

            const string& input_name = op->get_inputs().begin()->get_tensor().get_name();
            const Shape& input_shape = op->get_inputs().begin()->get_shape();

            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();
            const Shape& output_shape = op->get_outputs().begin()->get_shape();
            const element::Type& output_type =
                op->get_outputs().begin()->get_tensor().get_element_type();

            const shared_ptr<op::Sum> sum = static_pointer_cast<op::Sum>(op);
            const AxisSet& axis = sum->get_reduction_axes();

            if (axis.empty())
            {
                do_equal_propagation(topology, input_name, output_name);
            }
            else if (output_shape.empty())
            {
                do_bcast_sum_operation_scalar(topology,
                                              input_name,
                                              input_shape,
                                              output_name,
                                              output_shape,
                                              output_type,
                                              false);
            }
            else
            {
                do_bcast_sum_operation(topology,
                                       input_name,
                                       input_shape,
                                       output_name,
                                       output_shape,
                                       output_type,
                                       axis,
                                       false);
            }
        }
        else if ("Reshape" == op->description())
        {
            arguments_check(op, 1, 1);

            const string& input_name = op->get_inputs().begin()->get_tensor().get_name();
            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();
            const shared_ptr<op::Reshape> op_broadcast = static_pointer_cast<op::Reshape>(op);
            const AxisVector& broadcast_axes = op_broadcast->get_input_order();

            vector<uint16_t> permute_order({0, 1, 2, 3}); // No action by default
            const size_t max_dim = 4;
            const size_t scale =
                broadcast_axes.size() < max_dim ? max_dim - broadcast_axes.size() : 0;

            // Need to scale indexes up according on array rank.
            // For example, in 2D array, indexes are 0,1 but in 4D array it should be 2,3
            // because cldnn::tensor is always 4D assuming cldnn::bfyx model
            size_t rindex = max_dim;
            for (auto i = broadcast_axes.crbegin(); i != broadcast_axes.crend() && rindex > 0;
                 ++i, --rindex)
            {
                permute_order.at(rindex - 1) = *i + scale;
            }

            const cldnn::permute cldnn_permute(output_name, input_name, permute_order);
            topology.add(cldnn_permute);
        }
        else if ("Negative" == op->description())
        {
            const cldnn_activation_additional_params param = {-1.f, 0.f};
            do_unary_operation(topology, op, activation_linear, param);
        }
        else if ("Relu" == op->description())
        {
            do_unary_operation(topology, op, activation_relu);
        }
        else if ("ReluBackprop" == op->description())
        {
            arguments_check(op, 2, 1);

            const string& input = op->get_inputs().at(0).get_tensor().get_name();
            const string& input_grad = op->get_inputs().at(1).get_tensor().get_name();
            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();
            const cldnn_activation_additional_params& param = {0.f, 0.f};

            const cldnn::activation_grad cldnn_activ_grad(
                output_name, input_grad, input, activation_grad_relu, param);
            topology.add(cldnn_activ_grad);
        }
        else if ("Abs" == op->description())
        {
            do_unary_operation(topology, op, activation_abs);
        }
        else if ("Sqrt" == op->description())
        {
            do_unary_operation(topology, op, activation_sqrt);
        }
        else if ("Tanh" == op->description())
        {
            do_unary_operation(topology, op, activation_hyperbolic_tan);
        }
        else if ("Sigmoid" == op->description())
        {
            do_unary_operation(topology, op, activation_logistic);
        }
        else if ("Subtract" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::sub);
        }
        else if ("Power" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::pow);
        }
        else if ("Pad" == op->description())
        {
            arguments_check(op, 2, 1);

            const string& input_name = op->get_inputs().at(0).get_tensor().get_name();
            const Shape& input_shape = op->get_inputs().at(0).get_shape();
            const string& scalar_name = op->get_inputs().at(1).get_tensor().get_name();
            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();
            const Shape& output_shape = op->get_outputs().begin()->get_shape();
            const element::Type& output_type =
                op->get_outputs().begin()->get_tensor().get_element_type();

            const shared_ptr<op::Pad> pad = static_pointer_cast<op::Pad>(op);
            const Shape& pad_above = pad->get_padding_above();
            const Shape& pad_below = pad->get_padding_below();
            const Shape& pad_interior = pad->get_padding_interior();

            do_pad_operation(topology,
                             input_name,
                             input_shape,
                             scalar_name,
                             output_name,
                             output_shape,
                             output_type,
                             pad_below,
                             pad_interior);
        }
        else if ("BatchNorm" == op->description())
        {
            const shared_ptr<op::BatchNorm> batch_norm = static_pointer_cast<op::BatchNorm>(op);
            const double eps = batch_norm->get_eps_value();

            if (op->get_inputs().size() < 3 || op->get_outputs().empty())
            {
                arguments_check(op, 3, 1); // throw exception in this case
            }

            const string& output_name = op->get_outputs().begin()->get_tensor().get_name();
            const string& gamma_name = op->get_inputs().at(0).get_tensor().get_name();
            const string& beta_name = op->get_inputs().at(1).get_tensor().get_name();
            const string& input_name = op->get_inputs().at(2).get_tensor().get_name();
            const Shape& input_shape = op->get_inputs().at(2).get_shape();

            if (op->get_outputs().size() == 1)
            {
                arguments_check(op, 5, 1);

                const string& mean_name = op->get_inputs().at(3).get_tensor().get_name();
                const string& variance_name = op->get_inputs().at(4).get_tensor().get_name();

                do_batch_norm_operation(topology,
                                        output_name,
                                        eps,
                                        input_name,
                                        input_shape,
                                        gamma_name,
                                        beta_name,
                                        mean_name,
                                        variance_name);
            }
            else if (op->get_outputs().size() == 3)
            {
                arguments_check(op, 3, 3);

                do_batch_norm_operation(
                    topology, output_name, eps, input_name, input_shape, gamma_name, beta_name);
            }
            else
            {
                arguments_check(op, 5, 1); // throw exception in this case
            }
        }
        else if ("Convolution" == op->description())
        {
            arguments_check(op, 2, 1);

            const std::string& conv_name = op->get_outputs().begin()->get_tensor().get_name();
            const std::string& image_name = op->get_inputs().at(0).get_tensor().get_name();
            const std::string& weight_name = op->get_inputs().at(1).get_tensor().get_name();

            const shared_ptr<op::Convolution> conv_op = static_pointer_cast<op::Convolution>(op);

            const Strides& conv_stride = conv_op->get_window_movement_strides();
            const Strides& conv_dilation = conv_op->get_window_dilation_strides();
            const CoordinateDiff& conv_padding_below = conv_op->get_padding_below();
            const CoordinateDiff& conv_padding_above = conv_op->get_padding_above();
            const Strides& conv_data_dilation = conv_op->get_data_dilation_strides();

            if (conv_stride.size() > 2)
            {
                ostringstream os;
                os << "Unsupported strides for \"" << op->description() << '\"';
                throw std::invalid_argument(os.str());
            }

            if (conv_padding_below.size() > 2 || conv_padding_above.size() > 2)
            {
                ostringstream os;
                os << "Unsupported padding for \"" << op->description() << '\"';
                throw std::invalid_argument(os.str());
            }

            //TODO: Further clDNN version will work with different paddings above and below
            if (conv_padding_below.at(0) != conv_padding_above.at(0) ||
                conv_padding_below.at(1) != conv_padding_above.at(1))
            {
                ostringstream os;
                os << "Paddings above and below are different for \"" << op->description() << '\"';
                throw std::invalid_argument(os.str());
            }

            if (conv_dilation.size() > 2)
            {
                ostringstream os;
                os << "Unsupported dilation for \"" << op->description() << '\"';
                throw std::invalid_argument(os.str());
            }

            if (conv_data_dilation.size() > 2 || conv_data_dilation.at(0) != 1 ||
                conv_data_dilation.at(1) != 1)
            {
                ostringstream os;
                os << "Unsupported data dilation for \"" << op->description() << '\"';
                throw std::invalid_argument(os.str());
            }

            const cldnn::tensor input_offset(
                0, 0, -conv_padding_below.at(1), -conv_padding_below.at(0));
            const cldnn::tensor strides(1, 1, conv_stride.at(1), conv_stride.at(0));
            const cldnn::tensor dilation(1, 1, conv_dilation.at(1), conv_dilation.at(0));

            const cldnn::convolution cldnn_conv(
                conv_name, image_name, {weight_name}, strides, input_offset, dilation);
            topology.add(cldnn_conv);
        }
        else
        {
            ostringstream os;
            os << "Unsupported operation \"" << op->description() << '\"';
            throw invalid_argument(os.str());
        }
    }

    instance.ocl_network = make_shared<cldnn::network>(*ocl_engine, topology);

    return true;
}

bool runtime::intelgpu::IntelGPUBackend::call(
    shared_ptr<Function> func,
    const vector<shared_ptr<runtime::TensorView>>& outputs,
    const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    validate_call(func, outputs, inputs);

    FunctionInstance& instance = ocl_networks[func];
    if (instance.ocl_network == nullptr)
    {
        if (!compile(func))
        {
            return false;
        }
    }

    shared_ptr<cldnn::network> network = instance.ocl_network;

    // Process input parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_parameters and inputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < inputs.size(); i++)
    {
        shared_ptr<runtime::intelgpu::IntelGPUTensorView> tv =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(inputs[i]);
        const op::ParameterVector& input_params = func->get_parameters();
        const string& tensor_name = input_params[i]->get_output_tensor().get_name();
        network->set_input_data(tensor_name, *tv->get_data_ptr());
    }

    // Execute network
    map<cldnn::primitive_id, cldnn::network_output> result = network->execute();

    // Process output parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_results and outputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < func->get_output_size(); i++)
    {
        shared_ptr<runtime::intelgpu::IntelGPUTensorView> ngraph_res =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(outputs[i]);
        const string& tensor_name = func->get_output_op(i)->get_output_tensor().get_name();

        auto result_memory = result.at(tensor_name).get_memory().pointer<char>();
        ngraph_res->write(result_memory.data(), 0, result_memory.size());
    }

    return true;
}
