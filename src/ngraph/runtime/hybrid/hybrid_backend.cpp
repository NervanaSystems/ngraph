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

#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/assign_placement.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/cpu/cpu_placement.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/runtime/interpreter/int_placement.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::hybrid::HYBRIDBackend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

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

shared_ptr<runtime::Backend> runtime::hybrid::HYBRIDBackend::get_cached_backend(Placement placement)
{
    if (m_cached_backends.find(placement) == m_cached_backends.end())
    {
        m_cached_backends[placement] = runtime::Backend::create(placement_to_string(placement));
    }
    return m_cached_backends.at(placement);
}

shared_ptr<runtime::Tensor> runtime::hybrid::HYBRIDBackend::create_tensor(const element::Type& type,
                                                                          const Shape& shape)
{
    return make_shared<runtime::HostTensor>(type, shape, "external");
}

shared_ptr<runtime::Tensor> runtime::hybrid::HYBRIDBackend::create_tensor(const element::Type& type,
                                                                          const Shape& shape,
                                                                          void* memory_pointer)
{
    return make_shared<runtime::HostTensor>(type, shape, memory_pointer, "external");
}

bool runtime::hybrid::HYBRIDBackend::compile(shared_ptr<Function> function)
{
    NGRAPH_INFO << "hybrid compile -Begin ";
    if (m_function_map.find(function) == m_function_map.end())
    {
        // Clone function
        FunctionInstance instance;
        instance.m_function = clone_function(*function);

        pass::Manager pass_manager;

        // fall back to CPU as the base transformer
        // pass_manager.register_pass<pass::AssignPlacement>(
        //     runtime::interpreter::default_placement_policy);

        // fall back to Interpreter as the base transformer
        pass_manager.register_pass<pass::AssignPlacement>(runtime::cpu::default_placement_policy);

        pass_manager.run_passes(instance.m_function);

        NGRAPH_INFO << "hybrid compile -begin split  ";
        // Split function to sub_functions
        tie(instance.m_sub_functions, instance.m_map_parameter_to_result) =
            split_function_by_placement(instance.m_function);
        NGRAPH_INFO << "hybrid compile -End split  ";

        m_function_map.insert({function, instance});
        NGRAPH_INFO << "hybrid compile -map incertion successful";

        // Compile subfunctions in corresponding backends
        for (shared_ptr<Function>& sub_function : instance.m_sub_functions)
        {
            Placement placement = get_colocated_function_placement(sub_function);
            auto backend = get_cached_backend(placement);
            backend->compile(sub_function);
        }
    }
    NGRAPH_INFO << "hybrid compile -End ";
    return true;
}

bool runtime::hybrid::HYBRIDBackend::call(shared_ptr<Function> function,
                                          const vector<shared_ptr<runtime::Tensor>>& outputs,
                                          const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    NGRAPH_INFO << "hybrid call -Begin ";

    validate_call(function, outputs, inputs);

    // Get FunctionInstance
    bool rc = true;
    auto it = m_function_map.find(function);
    if (it == m_function_map.end())
    {
        compile(function);
        it = m_function_map.find(function);
    }
    if (it == m_function_map.end())
    {
        throw runtime_error("Error constructing backend.");
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
        Placement placement = get_colocated_function_placement(sub_function);
        auto backend = get_cached_backend(placement);

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
    NGRAPH_INFO << "hybrid call -End ";
    return rc;
}
