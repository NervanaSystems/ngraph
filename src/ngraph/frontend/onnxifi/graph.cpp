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

#include <onnxifi.h>

#include "backend.hpp"
#include "graph.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        bool Graph::run_graph()
        {
            ::onnxStatus status{::onnxWaitEvent(m_input_fence->event)};
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            bool result{m_backend.call(m_function, m_inputs, m_outputs)};
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
                status = ::onnxInitEvent(m_backend.get_handle(), &output_fence->event);
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

        bool Graph::compile() { return m_backend.compile(m_function); }

    } // namespace onnxifi

} // namespace ngraph
