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

#include <mutex>

#include <onnxifi.h>

#include "event.hpp"
#include "event_manager.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        ::onnxEvent EventManager::acquire()
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            std::unique_ptr<Event> event{new Event};
            auto pair = m_registered_events.emplace(reinterpret_cast<::onnxEvent>(event.get()), std::move(event));
            if (!pair.second)
            {
                throw status::no_system_resources{};
            }
            return (pair.first)->first;
        }

        void EventManager::release(::onnxEvent event)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            m_registered_events.erase(event);

        }

        const Event& EventManager::get_by_handle(::onnxEvent event) const
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            return *m_registered_events.at(event);
        }

    } // namespace onnxifi

} // namespace ngraph
