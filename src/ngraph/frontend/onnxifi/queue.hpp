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

#include <cstddef>
#include <mutex>
#include <queue>

namespace ngraph
{
    namespace onnxifi
    {
        template <typename T>
        class Queue
        {
        public:
            using value_type = T;
            using container_type = std::queue<value_type>;
            using reference = value_type&;
            using const_reference = const value_type&;
            using size_type = std::size_t;

            Queue(const Queue<T>& other, const std::lock_guard<std::mutex>&)
                : m_queue{other.m_queue}
            {
            }
            Queue(const Queue<T>& other)
                : Queue{other, std::lock_guard<std::mutex>{other.m_mutex}}
            {
            }

            Queue(Queue<T>&& other, const std::lock_guard<std::mutex>&) noexcept
                : m_queue{std::forward<Queue<T>>(other)}
            {
            }

            Queue(Queue<T>&& other) noexcept
                : Queue{std::forward<Queue<T>>(other), std::lock_guard<std::mutex>{other.m_mutex}}
            {
            }

            Queue() = default;

            Queue<T>& operator=(const Queue<T>& other)
            {
                if (&other != this)
                {
                    std::unique_lock<std::mutex> lock{m_mutex, std::defer_lock};
                    std::unique_lock<std::mutex> other_lock{m_mutex, std::defer_lock};
                    std::lock(lock, other_lock);
                    m_queue = other.m_queue;
                }
                return *this;
            }

            Queue<T>& operator=(Queue<T>&& other) noexcept
            {
                if (&other != this)
                {
                    std::unique_lock<std::mutex> lock{m_mutex, std::defer_lock};
                    std::unique_lock<std::mutex> other_lock{m_mutex, std::defer_lock};
                    std::lock(lock, other_lock);
                    m_queue = std::move(other.m_queue);
                }
                return *this;
            }

            void push(const_reference element)
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                m_queue.push(element);
            }

            template <typename... Args>
            void emplace(Args&&... args)
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                m_queue.emplace(std::forward<Args>(args)...);
            }

            reference back()
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return m_queue.back();
            }

            const_reference back() const
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return m_queue.back();
            }

            reference front()
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return m_queue.front();
            }

            const_reference front() const
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return m_queue.front();
            }

            void pop()
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                m_queue.pop();
            }

            bool empty() const
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return m_queue.empty();
            }

            void swap(Queue<T>& other)
            {
                if (&other != this)
                {
                    std::unique_lock<std::mutex> lock{m_mutex, std::defer_lock};
                    std::unique_lock<std::mutex> other_lock{other.m_mutex, std::defer_lock};
                    std::swap(m_queue, other.m_queue);
                }
            }

        private:
            mutable std::mutex m_mutex{};
            std::queue<T> m_queue{};

            reference get()
            {
                reference result{m_queue.front()};
                m_queue.pop();
                return result;
            }
        };
    }
}
