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

#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/hybrid/hybrid_util.hpp"
#include "ngraph/runtime/hybrid/pass/assign_placement.hpp"
#include "ngraph/runtime/tensor.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
void copy_data(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

template <typename T>
std::vector<T> read_vector(std::shared_ptr<ngraph::runtime::Tensor> tv)
{
    if (ngraph::element::from<T>() != tv->get_tensor_layout()->get_element_type())
    {
        throw std::invalid_argument("read_vector type must match Tensor type");
    }
    size_t element_count = ngraph::shape_size(tv->get_shape());
    size_t size = element_count * sizeof(T);
    std::vector<T> rc(element_count);
    tv->read(rc.data(), 0, size);
    return rc;
}

runtime::hybrid::HybridBackend::HybridBackend(
    const std::vector<std::pair<std::string, std::shared_ptr<runtime::Backend>>>& backend_list)
    : m_backend_list{backend_list}
{
}

shared_ptr<runtime::Tensor>
    runtime::hybrid::HybridBackend::create_tensor(const element::Type& element_type,
                                                  const Shape& shape)
{
    auto it = m_backend_list.begin();
    return it->second->create_tensor(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::hybrid::HybridBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    auto it = m_backend_list.begin();
    return it->second->create_tensor(element_type, shape, memory_pointer);
}

runtime::Handle runtime::hybrid::HybridBackend::compile(shared_ptr<Function> func)
{
    if (m_function_map.find(func) == m_function_map.end())
    {
        vector<shared_ptr<runtime::Backend>> backend_list;
        for (auto p : m_backend_list)
        {
            backend_list.push_back(p.second);
        }

        // Clone function
        FunctionInstance instance;
        instance.m_function = clone_function(*func);

        // Run placement pass
        ngraph::pass::Manager pass_manager;
        pass_manager.register_pass<runtime::hybrid::pass::AssignPlacement>(backend_list);
        pass_manager.run_passes(instance.m_function);

        // Split function to sub_functions
        tie(instance.m_sub_functions, instance.m_map_parameter_to_result) =
            split_function_by_placement_size(instance.m_function);
        m_function_map.insert({func, instance});

        // Compile subfunctions in corresponding backends
        for (shared_ptr<Function>& sub_function : instance.m_sub_functions)
        {
            size_t placement = get_colocated_function_placement_size(sub_function);
            auto backend = m_backend_list[placement];
            backend.second->compile(sub_function);

            // Compile will replace nodes so we need to make one more pass through all
            // ops to reset placement
            for (auto op : sub_function->get_ops())
            {
                op->set_placement_index(placement);
            }
        }
    }

    return func;
}

bool runtime::hybrid::HybridBackend::call(shared_ptr<Function> func,
                                          const vector<shared_ptr<runtime::Tensor>>& outputs,
                                          const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // Get FunctionInstance
    bool rc = true;

    auto it = m_function_map.find(func);
    if (it == m_function_map.end())
    {
        throw runtime_error("compile() must be called before call().");
    }
    FunctionInstance& instance = it->second;

    // Parameter and result node in sub_function maps to one Tensor
    unordered_map<shared_ptr<Node>, shared_ptr<runtime::Tensor>> map_node_to_tensor_view;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        map_node_to_tensor_view[instance.m_function->get_parameters()[i]] = inputs[i];
    }
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        map_node_to_tensor_view[instance.m_function->get_results()[i]] = outputs[i];
    }

    // Call subfunctions
    for (shared_ptr<Function>& sub_function : instance.m_sub_functions)
    {
        // Init backend
        size_t placement = get_colocated_function_placement_size(sub_function);
        auto backend = m_backend_list[placement].second;

        // Prepare parameter TensorViews
        vector<shared_ptr<runtime::Tensor>> parameter_tvs;
        for (auto parameter_node : sub_function->get_parameters())
        {
            if (map_node_to_tensor_view.find(parameter_node) != map_node_to_tensor_view.end())
            {
                parameter_tvs.push_back(map_node_to_tensor_view.at(parameter_node));
            }
            else
            {
                auto result_node = instance.m_map_parameter_to_result.at(parameter_node);
                auto result_tv = map_node_to_tensor_view.at(result_node);
                auto parameter_tv = backend->create_tensor(parameter_node->get_element_type(),
                                                           parameter_node->get_shape());
                copy_data(parameter_tv, read_vector<float>(result_tv));
                map_node_to_tensor_view[parameter_node] = parameter_tv;
                parameter_tvs.push_back(parameter_tv);
            }
        }

        // Prepare result TensorViews
        vector<shared_ptr<runtime::Tensor>> result_tvs;
        for (auto result_node : sub_function->get_results())
        {
            if (map_node_to_tensor_view.find(result_node) != map_node_to_tensor_view.end())
            {
                result_tvs.push_back(map_node_to_tensor_view.at(result_node));
            }
            else
            {
                auto result_tv = backend->create_tensor(result_node->get_element_type(),
                                                        result_node->get_shape());
                map_node_to_tensor_view[result_node] = result_tv;
                result_tvs.push_back(result_tv);
            }
        }

        // Call
        backend->call_with_validate(sub_function, result_tvs, parameter_tvs);
    }
    return rc;
}

bool runtime::hybrid::HybridBackend::is_supported(const Node& node) const
{
    return true;
}
