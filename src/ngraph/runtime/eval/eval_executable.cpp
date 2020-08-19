//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/runtime/eval/eval_executable.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/implicit_broadcast_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

runtime::eval::EVALExecutable::EVALExecutable(const shared_ptr<Function>& function,
                                              bool enable_performance_collection)
{
    m_function = clone_function(*function);

    auto is_supported = [](const Node& node) {
        bool retval = false;
        switch (EVALExecutable::get_typeid(node))
        {
        case OP_TYPEID::Clamp_v0:
        case OP_TYPEID::MatMul_v0:
        case OP_TYPEID::Mod_v1:
        case OP_TYPEID::Squeeze_v0:
        case OP_TYPEID::Unsqueeze_v0: retval = true; break;
        default: break;
        }
        return retval;
    };
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::LikeReplacement>();
    pass_manager.register_pass<pass::FusedOpDecomposition>(is_supported);
    pass_manager.run_passes(m_function);
    for (auto node : m_function->get_ordered_ops())
    {
        m_nodes.push_back(node);
    }
    set_parameters_and_results(*m_function);
}

bool runtime::eval::EVALExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                         const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // convert inputs to HostTensor
    vector<shared_ptr<HostTensor>> func_inputs;
    for (auto tensor : inputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_inputs.push_back(host_tensor);
    }

    // convert outputs to HostTensor
    vector<shared_ptr<HostTensor>> func_outputs;
    for (auto tensor : outputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_outputs.push_back(host_tensor);
    }

    // map function params -> HostTensor
    unordered_map<descriptor::Tensor*, shared_ptr<HostTensor>> tensor_map;
    size_t input_count = 0;
    for (auto param : get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &param->output(i).get_tensor();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    // map function outputs -> HostTensor
    for (size_t output_count = 0; output_count < get_results().size(); ++output_count)
    {
        auto output = get_results()[output_count];
        if (!is_type<op::v0::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::v0::Result");
        }
        descriptor::Tensor* tensor = &output->get_output_tensor(0);
        tensor_map.insert({tensor, func_outputs[output_count]});
    }

    // for each ordered op in the graph
    for (auto& op : m_nodes)
    {
        auto type_id = get_typeid(*op);
        if (type_id == OP_TYPEID::Parameter_v0)
        {
            continue;
        }

        // get op inputs from map
        vector<shared_ptr<HostTensor>> op_inputs;
        for (auto input : op->inputs())
        {
            descriptor::Tensor* tensor = &input.get_tensor();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        // get op outputs from map or create
        vector<shared_ptr<HostTensor>> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &op->output(i).get_tensor();
            shared_ptr<HostTensor> host_tensor;
            auto it = tensor_map.find(tensor);
            if (it == tensor_map.end())
            {
                const Shape& shape = op->get_output_shape(i);
                const element::Type& type = op->get_output_element_type(i);
                string name = op->output(i).get_tensor().get_name();
                host_tensor = make_shared<runtime::HostTensor>(type, shape, name);
                tensor_map.insert({tensor, host_tensor});
            }
            else
            {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
        }

        string name = op->description() + "_v" + to_string(op->get_type_info().version);
        if (!op->evaluate(op_outputs, op_inputs))
        {
            throw unsupported_op("Unsupported op '" + name + "'");
        }
    }

    return true;
}

runtime::eval::OP_TYPEID runtime::eval::EVALExecutable::get_typeid(const Node& node)
{
    const NodeTypeInfo& type_info = node.get_type_info();
    // This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
    // {Abs::type_info, OP_TYPEID::Abs},
    // {Acos::type_info, OP_TYPEID::Acos},
    // ...
    static const map<NodeTypeInfo, OP_TYPEID> type_info_map{
#define NGRAPH_OP(NAME, VERSION)                                                                   \
    {ngraph::op::v##VERSION::NAME::type_info, OP_TYPEID::NAME##_v##VERSION},
#include "ngraph/op_version_tbl.hpp"
#undef NGRAPH_OP
    };
    OP_TYPEID rc = OP_TYPEID::UnknownOp;

    auto it = type_info_map.find(type_info);
    if (it != type_info_map.end())
    {
        rc = it->second;
    }
    return rc;
}
