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

#pragma once

#include <condition_variable>
#include <mutex>

#include <onnxifi.h>

#include "exceptions.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        namespace detail
        {
            /// \brief Implementation of onnxEvent data type.
            template <bool Autoreset>
            class Event
            {
            public:
                Event(const Event&) = delete;
                Event& operator=(const Event&) = delete;

                Event() = default;

                Event(Event&& other, std::lock_guard<std::mutex>&) noexcept
                    : m_signaled{other.m_signaled}
                {
                }

                Event(Event&& other) noexcept
                    : Event{other, std::lock_guard<std::mutex>{other.m_mutex}}
                {
                }

                Event& operator=(Event&& other) noexcept
                {
                    if (this != &other)
                    {
                        std::unique_lock<std::mutex> lock{m_mutex, std::defer_lock};
                        std::unique_lock<std::mutex> other_lock{other.m_mutex, std::defer_lock};
                        std::lock(lock, other_lock);
                        m_signaled = other.m_signaled;
                    }
                    return *this;
                }

                void signal()
                {
                    std::lock_guard<std::mutex> lock{m_mutex};
                    if (m_signaled)
                    {
                        throw status::invalid_state{};
                    }
                    m_signaled = true;
                    m_condition_variable.notify_all();
                }

                void reset()
                {
                    std::lock_guard<std::mutex> lock{m_mutex};
                    m_signaled = false;
                }

                void wait() const
                {
                    std::unique_lock<std::mutex> lock{m_mutex};
                    m_condition_variable.wait(lock, [&] { return m_signaled; });
                    if (Autoreset)
                    {
                        m_signaled = false;
                    }
                }

                template <typename Rep, typename Period>
                bool wait_for(const std::chrono::duration<Rep, Period>& duration) const
                {
                    std::unique_lock<std::mutex> lock{m_mutex};
                    auto result =
                        m_condition_variable.wait_for(lock, duration, [&] { return m_signaled; });
                    if (Autoreset)
                    {
                        m_signaled = false;
                    }
                    return result;
                }

                template <typename Clock, typename Duration>
                bool wait_until(const std::chrono::time_point<Clock, Duration>& time_point) const
                {
                    std::unique_lock<std::mutex> lock{m_mutex};
                    auto result = m_condition_variable.wait_until(
                        lock, time_point, [&] { return m_signaled; });
                    if (Autoreset)
                    {
                        m_signaled = false;
                    }
                    return result;
                }

                bool is_signaled() const
                {
                    std::lock_guard<std::mutex> lock{m_mutex};
                    return m_signaled;
                }

                void get_state(::onnxEventState* state)
                {
                    if (state == nullptr)
                    {
                        throw status::null_pointer{};
                    }
                    *state = is_signaled() ? ONNXIFI_EVENT_STATE_SIGNALLED
                                           : ONNXIFI_EVENT_STATE_NONSIGNALLED;
                }

            private:
                mutable std::mutex m_mutex{};
                mutable std::condition_variable m_condition_variable{};
                mutable bool m_signaled{false};
            };

        } // namespace detail

        using Event = detail::Event<false>;
        using EventAuto = detail::Event<true>;

    } // namespace onnxifi

} // namespace ngraph
