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

#include <mutex>

#include <onnxifi.h>

#include "event.hpp"
#include "event_manager.hpp"

#include "exceptions.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        void EventManager::init_event(::onnxBackend handle, ::onnxEvent* event)
        {
            if (event == nullptr)
            {
                throw status::null_pointer{};
            }
            auto& backend = BackendManager::get_backend(handle);
            *event = instance()._init_event(backend);
        }

        ::onnxEvent EventManager::_init_event(const Backend&)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            std::unique_ptr<Event> event{new Event};
            auto pair = m_registered_events.emplace(reinterpret_cast<::onnxEvent>(event.get()),
                                                    std::move(event));
            if (!pair.second)
            {
                throw status::no_system_resources{};
            }
            return (pair.first)->first;
        }

        void EventManager::_release_event(::onnxEvent event)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            m_registered_events.erase(event);
        }

    } // namespace onnxifi

} // namespace ngraph
