//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

#define ADD_TYPE(a)                                                                                \
    {                                                                                              \
        #a, runtime::interpreter::OP_TYPEID::a##_TYPEID                                            \
    }

static unordered_map<string, runtime::interpreter::OP_TYPEID> s_typeid_map{
    ADD_TYPE(Abs),
    ADD_TYPE(Acos),
    ADD_TYPE(Add),
    ADD_TYPE(AllReduce),
    ADD_TYPE(And),
    ADD_TYPE(ArgMax),
    ADD_TYPE(ArgMin),
    ADD_TYPE(Asin),
    ADD_TYPE(Atan),
    ADD_TYPE(AvgPool),
    ADD_TYPE(AvgPoolBackprop),
    ADD_TYPE(BatchNorm),
    ADD_TYPE(BatchNormBackprop),
    ADD_TYPE(Broadcast),
    ADD_TYPE(Ceiling),
    ADD_TYPE(Concat),
    ADD_TYPE(Constant),
    ADD_TYPE(Convert),
    ADD_TYPE(Convolution),
    ADD_TYPE(ConvolutionBackpropData),
    ADD_TYPE(ConvolutionBackpropFilters),
    ADD_TYPE(Cos),
    ADD_TYPE(Cosh),
    ADD_TYPE(Divide),
    ADD_TYPE(Dot),
    ADD_TYPE(Equal),
    ADD_TYPE(Exp),
    ADD_TYPE(Floor),
    ADD_TYPE(FunctionCall),
    ADD_TYPE(GetOutputElement),
    ADD_TYPE(Greater),
    ADD_TYPE(GreaterEq),
    ADD_TYPE(Less),
    ADD_TYPE(LessEq),
    ADD_TYPE(Log),
    ADD_TYPE(LRN),
    ADD_TYPE(Max),
    ADD_TYPE(Maximum),
    ADD_TYPE(MaxPool),
    ADD_TYPE(MaxPoolBackprop),
    ADD_TYPE(Min),
    ADD_TYPE(Minimum),
    ADD_TYPE(Multiply),
    ADD_TYPE(Negative),
    ADD_TYPE(Not),
    ADD_TYPE(NotEqual),
    ADD_TYPE(OneHot),
    ADD_TYPE(Or),
    ADD_TYPE(Pad),
    ADD_TYPE(Parameter),
    ADD_TYPE(Power),
    ADD_TYPE(Product),
    ADD_TYPE(Reduce),
    ADD_TYPE(ReduceWindow),
    ADD_TYPE(Relu),
    ADD_TYPE(ReluBackprop),
    ADD_TYPE(ReplaceSlice),
    ADD_TYPE(Reshape),
    ADD_TYPE(Result),
    ADD_TYPE(Reverse),
    ADD_TYPE(ReverseSequence),
    ADD_TYPE(Select),
    ADD_TYPE(SelectAndScatter),
    ADD_TYPE(Sigmoid),
    ADD_TYPE(SigmoidBackprop),
    ADD_TYPE(Sign),
    ADD_TYPE(Sin),
    ADD_TYPE(Sinh),
    ADD_TYPE(Slice),
    ADD_TYPE(Softmax),
    ADD_TYPE(Sqrt),
    ADD_TYPE(StopGradient),
    ADD_TYPE(Subtract),
    ADD_TYPE(Sum),
    ADD_TYPE(Tan),
    ADD_TYPE(Tanh)};

// {"Abs", runtime::interpreter::OP_TYPEID::Abs_TYPEID},
// {"Acos", runtime::interpreter::OP_TYPEID::Acos_TYPEID},
// {"Add", runtime::interpreter::OP_TYPEID::Add_TYPEID},
// {"AllReduce", runtime::interpreter::OP_TYPEID::AllReduce_TYPEID},
// {"And", runtime::interpreter::OP_TYPEID::And_TYPEID},
// {"ArgMax", runtime::interpreter::OP_TYPEID::ArgMax_TYPEID},
// {"ArgMin", runtime::interpreter::OP_TYPEID::ArgMin_TYPEID},
// {"Asin", runtime::interpreter::OP_TYPEID::Asin_TYPEID},
// {"Atan", runtime::interpreter::OP_TYPEID::Atan_TYPEID},
// {"AvgPool", runtime::interpreter::OP_TYPEID::AvgPool_TYPEID},
// {"BatchNorm", runtime::interpreter::OP_TYPEID::BatchNorm_TYPEID},
// {"Broadcast", runtime::interpreter::OP_TYPEID::Broadcast_TYPEID},
// {"Ceiling", runtime::interpreter::OP_TYPEID::Ceiling_TYPEID},
// {"Concat", runtime::interpreter::OP_TYPEID::Concat_TYPEID},
// {"Constant", runtime::interpreter::OP_TYPEID::Constant_TYPEID},
// {"Convert", runtime::interpreter::OP_TYPEID::Convert_TYPEID},
// {"Convolution", runtime::interpreter::OP_TYPEID::Convolution_TYPEID},
// {"Cos", runtime::interpreter::OP_TYPEID::Cos_TYPEID},
// {"Cosh", runtime::interpreter::OP_TYPEID::Cosh_TYPEID},
// {"Divide", runtime::interpreter::OP_TYPEID::Divide_TYPEID},
// {"Dot", runtime::interpreter::OP_TYPEID::Dot_TYPEID},
// {"Equal", runtime::interpreter::OP_TYPEID::Equal_TYPEID},
// {"Exp", runtime::interpreter::OP_TYPEID::Exp_TYPEID},
// {"Floor", runtime::interpreter::OP_TYPEID::Floor_TYPEID},
// {"FunctionCall", runtime::interpreter::OP_TYPEID::FunctionCall_TYPEID},
// {"GetOutputElement", runtime::interpreter::OP_TYPEID::GetOutputElement_TYPEID},
// {"Greater", runtime::interpreter::OP_TYPEID::Greater_TYPEID},
// {"GreaterEq", runtime::interpreter::OP_TYPEID::GreaterEq_TYPEID},
// {"Less", runtime::interpreter::OP_TYPEID::Less_TYPEID},
// {"LessEq", runtime::interpreter::OP_TYPEID::LessEq_TYPEID},
// {"Log", runtime::interpreter::OP_TYPEID::Log_TYPEID},
// {"Max", runtime::interpreter::OP_TYPEID::Max_TYPEID},
// {"Maximum", runtime::interpreter::OP_TYPEID::Maximum_TYPEID},
// {"MaxPool", runtime::interpreter::OP_TYPEID::MaxPool_TYPEID},
// {"Min", runtime::interpreter::OP_TYPEID::Min_TYPEID},
// {"Minimum", runtime::interpreter::OP_TYPEID::Minimum_TYPEID},
// {"Multiply", runtime::interpreter::OP_TYPEID::Multiply_TYPEID},
// {"Negative", runtime::interpreter::OP_TYPEID::Negative_TYPEID},
// {"Not", runtime::interpreter::OP_TYPEID::Not_TYPEID},
// {"NotEqual", runtime::interpreter::OP_TYPEID::NotEqual_TYPEID},
// {"OneHot", runtime::interpreter::OP_TYPEID::OneHot_TYPEID},
// {"Or", runtime::interpreter::OP_TYPEID::Or_TYPEID},
// {"Pad", runtime::interpreter::OP_TYPEID::Pad_TYPEID},
// {"Parameter", runtime::interpreter::OP_TYPEID::Parameter_TYPEID},
// {"Power", runtime::interpreter::OP_TYPEID::Power_TYPEID},
// {"Product", runtime::interpreter::OP_TYPEID::Product_TYPEID},
// {"Reduce", runtime::interpreter::OP_TYPEID::Reduce_TYPEID},
// {"ReduceWindow", runtime::interpreter::OP_TYPEID::ReduceWindow_TYPEID},
// {"Relu", runtime::interpreter::OP_TYPEID::Relu_TYPEID},
// {"ReluBackprop", runtime::interpreter::OP_TYPEID::ReluBackprop_TYPEID},
// {"ReplaceSlice", runtime::interpreter::OP_TYPEID::ReplaceSlice_TYPEID},
// {"Reshape", runtime::interpreter::OP_TYPEID::Reshape_TYPEID},
// {"Result", runtime::interpreter::OP_TYPEID::Result_TYPEID},
// {"Reverse", runtime::interpreter::OP_TYPEID::Reverse_TYPEID},
// {"ReverseSequence", runtime::interpreter::OP_TYPEID::ReverseSequence_TYPEID},
// {"Select", runtime::interpreter::OP_TYPEID::Select_TYPEID},
// {"SelectAndScatter", runtime::interpreter::OP_TYPEID::SelectAndScatter_TYPEID},
// {"Sigmoid", runtime::interpreter::OP_TYPEID::Sigmoid_TYPEID},
// {"SigmoidBackprop", runtime::interpreter::OP_TYPEID::SigmoidBackprop_TYPEID},
// {"Sign", runtime::interpreter::OP_TYPEID::Sign_TYPEID},
// {"Sin", runtime::interpreter::OP_TYPEID::Sin_TYPEID},
// {"Sinh", runtime::interpreter::OP_TYPEID::Sinh_TYPEID},
// {"Slice", runtime::interpreter::OP_TYPEID::Slice_TYPEID},
// {"Softmax", runtime::interpreter::OP_TYPEID::Softmax_TYPEID},
// {"Sqrt", runtime::interpreter::OP_TYPEID::Sqrt_TYPEID},
// {"StopGradient", runtime::interpreter::OP_TYPEID::StopGradient_TYPEID},
// {"Subtract", runtime::interpreter::OP_TYPEID::Subtract_TYPEID},
// {"Sum", runtime::interpreter::OP_TYPEID::Sum_TYPEID},
// {"Tan", runtime::interpreter::OP_TYPEID::Tan_TYPEID},
// {"Tanh", runtime::interpreter::OP_TYPEID::Tanh_TYPEID}};

using descriptor::layout::DenseTensorViewLayout;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::interpreter::INTBackend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

shared_ptr<runtime::TensorView>
    runtime::interpreter::INTBackend::create_tensor(const element::Type& type, const Shape& shape)
{
    return make_shared<runtime::HostTensorView>(type, shape, "external");
}

shared_ptr<runtime::TensorView> runtime::interpreter::INTBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::HostTensorView>(type, shape, memory_pointer, "external");
}

bool runtime::interpreter::INTBackend::compile(shared_ptr<Function> function)
{
    FunctionInstance& instance = m_function_map[function];
    if (!instance.m_is_compiled)
    {
        instance.m_is_compiled = true;
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::LikeReplacement>();
        pass_manager.register_pass<pass::AssignLayout<DenseTensorViewLayout>>();
        pass_manager.register_pass<pass::Liveness>();
        pass_manager.run_passes(function);

        for (const shared_ptr<Node>& node : function->get_ordered_ops())
        {
            auto it = s_typeid_map.find(node->description());
            if (it != s_typeid_map.end())
            {
                instance.m_wrapped_nodes.emplace_back(node, it->second);
            }
            else
            {
                // TODO: use unsupported_op when that is merged to master
                throw runtime_error(node->description());
            }
        }
    }

    return true;
}

bool runtime::interpreter::INTBackend::call(shared_ptr<Function> function,
                                            const vector<shared_ptr<runtime::TensorView>>& outputs,
                                            const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    validate_call(function, outputs, inputs);

    compile(function);
    FunctionInstance& instance = m_function_map[function];

    // convert inputs to HostTensorView
    vector<shared_ptr<runtime::HostTensorView>> func_inputs;
    for (auto tv : inputs)
    {
        func_inputs.push_back(static_pointer_cast<runtime::HostTensorView>(tv));
    }
    if (instance.m_nan_check_enabled)
    {
        perform_nan_check(func_inputs);
    }

    // convert outputs to HostTensorView
    vector<shared_ptr<runtime::HostTensorView>> func_outputs;
    for (auto tv : outputs)
    {
        func_outputs.push_back(static_pointer_cast<runtime::HostTensorView>(tv));
    }

    // map function params -> HostTensorView
    unordered_map<descriptor::TensorView*, shared_ptr<runtime::HostTensorView>> tensor_map;
    size_t input_count = 0;
    for (auto param : function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::TensorView* tv = param->get_output_tensor_view(i).get();
            tensor_map.insert({tv, func_inputs[input_count++]});
        }
    }

    // map function outputs -> HostTensorView
    for (size_t output_count = 0; output_count < function->get_output_size(); ++output_count)
    {
        auto output = function->get_output_op(output_count);
        if (!dynamic_pointer_cast<op::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::TensorView* tv = output->get_output_tensor_view(0).get();
        tensor_map.insert({tv, func_outputs[output_count]});
    }

    // for each ordered op in the graph
    for (const NodeWrapper& wrapped : instance.m_wrapped_nodes)
    {
        Node& op = wrapped.get_node();
        auto type_id = wrapped.get_typeid();
        if (op.description() == "Parameter")
        {
            continue;
        }
        // get op inputs from map
        vector<shared_ptr<runtime::HostTensorView>> op_inputs;
        for (const descriptor::Input& input : op.get_inputs())
        {
            descriptor::TensorView* tv = input.get_output().get_tensor_view().get();
            op_inputs.push_back(tensor_map.at(tv));
        }

        // get op outputs from map or create
        vector<shared_ptr<runtime::HostTensorView>> op_outputs;
        for (size_t i = 0; i < op.get_output_size(); ++i)
        {
            descriptor::TensorView* tv = op.get_output_tensor_view(i).get();
            shared_ptr<runtime::HostTensorView> htv;
            if (!contains_key(tensor_map, tv))
            {
                // the output tensor is not in the tensor map so create a new tensor
                const Shape& shape = op.get_output_shape(i);
                const element::Type& type = op.get_output_element_type(i);
                string name = op.get_output_tensor(i).get_name();
                htv = make_shared<runtime::HostTensorView>(type, shape, name);
                tensor_map.insert({tv, htv});
            }
            else
            {
                htv = tensor_map.at(tv);
            }
            op_outputs.push_back(htv);
        }

        // get op type
        element::Type type;
        switch (type_id)
        {
        case OP_TYPEID::Convert_TYPEID:
            type = op.get_inputs().at(0).get_tensor().get_element_type();
            break;
        case OP_TYPEID::Equal_TYPEID:
        case OP_TYPEID::Greater_TYPEID:
        case OP_TYPEID::GreaterEq_TYPEID:
        case OP_TYPEID::Less_TYPEID:
        case OP_TYPEID::LessEq_TYPEID:
        case OP_TYPEID::NotEqual_TYPEID:
            // Get the type of the second input, not the first
            // All BinaryElementwiseComparision ops have the same type for inputs
            // Select has bool for first input and the type we are interested in for the second
            type = op.get_inputs().at(1).get_tensor().get_element_type();
            break;
        default: type = op.get_outputs().at(0).get_element_type(); break;
        }

        if (instance.m_performance_counters_enabled)
        {
            instance.m_timer_map[&op].start();
        }
        generate_calls(type, wrapped, op_outputs, op_inputs);
        if (instance.m_performance_counters_enabled)
        {
            instance.m_timer_map[&op].stop();
        }
        if (instance.m_nan_check_enabled)
        {
            perform_nan_check(op_outputs, &op);
        }

        // delete any obsolete tensors
        for (const descriptor::Tensor* t : op.liveness_free_list)
        {
            for (auto it = tensor_map.begin(); it != tensor_map.end(); ++it)
            {
                if (it->second->get_tensor().get_name() == t->get_name())
                {
                    tensor_map.erase(it);
                    break;
                }
            }
        }
    }

    return true;
}

void runtime::interpreter::INTBackend::generate_calls(
    const element::Type& type,
    const NodeWrapper& op,
    const vector<shared_ptr<HostTensorView>>& outputs,
    const vector<shared_ptr<HostTensorView>>& inputs)
{
    if (type == element::boolean)
    {
        op_engine<char>(op, outputs, inputs);
    }
    else if (type == element::f32)
    {
        op_engine<float>(op, outputs, inputs);
    }
    else if (type == element::f64)
    {
        op_engine<double>(op, outputs, inputs);
    }
    else if (type == element::i8)
    {
        op_engine<int8_t>(op, outputs, inputs);
    }
    else if (type == element::i16)
    {
        op_engine<int16_t>(op, outputs, inputs);
    }
    else if (type == element::i32)
    {
        op_engine<int32_t>(op, outputs, inputs);
    }
    else if (type == element::i64)
    {
        op_engine<int64_t>(op, outputs, inputs);
    }
    else if (type == element::u8)
    {
        op_engine<uint8_t>(op, outputs, inputs);
    }
    else if (type == element::u16)
    {
        op_engine<uint16_t>(op, outputs, inputs);
    }
    else if (type == element::u32)
    {
        op_engine<uint32_t>(op, outputs, inputs);
    }
    else if (type == element::u64)
    {
        op_engine<uint64_t>(op, outputs, inputs);
    }
    else
    {
        stringstream ss;
        ss << "unsupported element type " << type << " op " << op.get_node().get_name();
        throw ngraph_error(ss.str());
    }
}

void runtime::interpreter::INTBackend::set_nan_check(shared_ptr<Function> func, bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    instance.m_nan_check_enabled = enable;
}

void runtime::interpreter::INTBackend::enable_performance_data(shared_ptr<Function> func,
                                                               bool enable)
{
    FunctionInstance& instance = m_function_map[func];
    instance.m_performance_counters_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::interpreter::INTBackend::get_performance_data(shared_ptr<Function> func) const
{
    vector<runtime::PerformanceCounter> rc;
    const FunctionInstance& instance = m_function_map.at(func);
    for (const pair<const Node*, stopwatch> p : instance.m_timer_map)
    {
        rc.emplace_back(p.first->get_name().c_str(),
                        p.second.get_total_microseconds(),
                        p.second.get_call_count());
    }
    return rc;
}

void runtime::interpreter::INTBackend::perform_nan_check(
    const vector<shared_ptr<HostTensorView>>& tvs, const Node* op)
{
    size_t arg_number = 1;
    for (shared_ptr<HostTensorView> tv : tvs)
    {
        const element::Type& type = tv->get_tensor().get_element_type();
        if (type == element::f32)
        {
            const float* data = tv->get_data_ptr<float>();
            for (size_t i = 0; i < tv->get_element_count(); i++)
            {
                if (std::isnan(data[i]))
                {
                    if (op)
                    {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    }
                    else
                    {
                        throw runtime_error("nan found in function's input tensor number " +
                                            to_string(arg_number));
                    }
                }
            }
        }
        else if (type == element::f64)
        {
            const double* data = tv->get_data_ptr<double>();
            for (size_t i = 0; i < tv->get_element_count(); i++)
            {
                if (std::isnan(data[i]))
                {
                    if (op)
                    {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    }
                    else
                    {
                        throw runtime_error("nan found in function's input tensor number " +
                                            to_string(arg_number));
                    }
                }
            }
        }
        arg_number++;
    }
}
