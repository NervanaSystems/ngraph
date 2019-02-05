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

#include <onnxifi.h>

#include "backend.hpp"
#include "graph.hpp"
#include "weight.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        Graph::~Graph() { m_backend->remove_compiled_function(m_handle); }
        bool Graph::run_graph()
        {
            ::onnxStatus status{::onnxWaitEvent(m_input_fence->event)};
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            bool result{m_backend->call(m_handle, m_ng_inputs, m_ng_outputs)};
            from_ng_outputs(m_ng_outputs, m_outputs);
            status = ::onnxSignalEvent(m_output_fence->event);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            return result;
        }

        void Graph::configure_memory_fences(const ::onnxMemoryFenceV1* input_fence,
                                            ::onnxMemoryFenceV1* output_fence)
        {
            if ((input_fence == nullptr) || (output_fence == nullptr))
            {
                throw status::null_pointer{};
            }
            if ((input_fence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1) ||
                (output_fence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1))
            {
                throw status::unsupported_tag{};
            }
            if ((input_fence->type == ONNXIFI_SYNCHRONIZATION_IMPLICIT) ||
                (output_fence->type == ONNXIFI_SYNCHRONIZATION_IMPLICIT))
            {
                throw status::unsupported_fence_type{};
            }
            if ((input_fence->type != ONNXIFI_SYNCHRONIZATION_EVENT) ||
                (output_fence->type != ONNXIFI_SYNCHRONIZATION_EVENT))
            {
                throw status::invalid_fence_type{};
            }
            ::onnxEventState state;
            ::onnxStatus status{::onnxGetEventState(output_fence->event, &state)};
            if (status == ONNXIFI_STATUS_INVALID_EVENT)
            {
                status = ::onnxInitEvent(m_backend->get_handle(), &output_fence->event);
                if (status != ONNXIFI_STATUS_SUCCESS)
                {
                    throw status::runtime{status};
                }
                status = ::onnxGetEventState(output_fence->event, &state);
            }
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            if (state != ONNXIFI_EVENT_STATE_NONSIGNALLED)
            {
                throw status::invalid_state{};
            }
            status = ::onnxGetEventState(input_fence->event, &state);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            m_input_fence = input_fence;
            m_output_fence = output_fence;
        }

        void Graph::load(std::istream& sin,
                         const Span<::onnxTensorDescriptorV1>& weight_descriptors)
        {
            std::unordered_map<std::string, onnx_import::Weight> weights;
            if (weight_descriptors.data() != nullptr)
            {
                if (weight_descriptors.empty())
                {
                    throw status::invalid_size{};
                }
                for (const auto& weight : weight_descriptors)
                {
                    Weight t{weight};
                    weights.emplace(t.name(), t.get());
                }
            }
            else
            {
                if (!weight_descriptors.empty())
                {
                    throw status::null_pointer{};
                }
            }
            auto function = onnx_import::import_onnx_model(sin, weights);
            m_handle = m_backend->compile(function);
        }

        void Graph::set_inputs(const Span<::onnxTensorDescriptorV1>& inputs)
        {
            if ((inputs.data() != nullptr) && inputs.empty())
            {
                throw status::invalid_size{};
            }
            if (inputs.is_valid())
            {
                m_ng_inputs.clear();
                for (const auto& descriptor : inputs)
                {
                    Tensor tensor{descriptor};
                    m_ng_inputs.emplace_back(tensor.to_ng(m_backend->get_backend()));
                }
            }
        }

        void Graph::set_outputs(const Span<::onnxTensorDescriptorV1>& outputs)
        {
            if (outputs.data() == nullptr)
            {
                throw status::null_pointer{};
            }
            if (outputs.empty())
            {
                throw status::invalid_size{};
            }
            m_ng_outputs.clear();
            m_outputs.clear();
            for (const auto& descriptor : outputs)
            {
                m_outputs.emplace_back(descriptor);
                m_ng_outputs.emplace_back(m_outputs.back().to_ng(m_backend->get_backend()));
            }
        }

    } // namespace onnxifi

} // namespace ngraph
