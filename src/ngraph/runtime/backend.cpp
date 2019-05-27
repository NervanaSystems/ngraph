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

#include <sstream>

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/dynamic/dynamic_backend.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::Backend::~Backend()
{
    NGRAPH_INFO;
    async_thread_stop();
    NGRAPH_INFO;
}

std::shared_ptr<ngraph::Node> runtime::Backend::get_backend_op(const std::string& op_name, ...)
{
    std::shared_ptr<ngraph::Node> dummy_node(nullptr);
    return dummy_node;
}

std::shared_ptr<runtime::Backend> runtime::Backend::create(const string& type,
                                                           bool must_support_dynamic)
{
    auto inner_backend = BackendManager::create_backend(type);

    if (!must_support_dynamic || inner_backend->supports_dynamic_tensors())
    {
        return inner_backend;
    }
    else
    {
        return make_shared<runtime::dynamic::DynamicBackend>(inner_backend);
    }
}

vector<string> runtime::Backend::get_registered_devices()
{
    return BackendManager::get_registered_backends();
}

std::shared_ptr<ngraph::runtime::Tensor>
    runtime::Backend::create_dynamic_tensor(const ngraph::element::Type& element_type,
                                            const PartialShape& shape)
{
    throw std::invalid_argument("This backend does not support dynamic tensors");
}

std::shared_ptr<runtime::Executable>
    runtime::Backend::compile(std::shared_ptr<Function> func,
                              ngraph::pass::PassConfig& pass_config,
                              bool enable_performance_data)
{
    return compile(func, enable_performance_data);
}

bool runtime::Backend::is_supported(const Node& node) const
{
    // The default behavior is that a backend does not support any ops. If this is not the case
    // then override this method and enhance.
    return false;
}

bool runtime::Backend::is_supported_property(const Property prop) const
{
    return false;
}

void runtime::Backend::remove_compiled_function(std::shared_ptr<Executable> exec)
{
}

std::shared_ptr<runtime::Executable> runtime::Backend::load(istream& input_stream)
{
    throw runtime_error("load opertion unimplemented.");
}

runtime::Backend::AsyncEvent::AsyncEvent(Type type,
                                         size_t buffer_number,
                                         void* p,
                                         size_t size_in_bytes)
    : m_type{type}
    , m_buffer_number{buffer_number}
    , m_data{p}
    , m_size_in_bytes{size_in_bytes}
    , m_executable{nullptr}
    , m_outputs{nullptr}
    , m_inputs{nullptr}
{
    (void)m_buffer_number;
}

runtime::Backend::AsyncEvent::AsyncEvent(size_t buffer_number,
                                         const shared_ptr<Executable>& executable,
                                         const vector<shared_ptr<runtime::Tensor>>& outputs,
                                         const vector<shared_ptr<runtime::Tensor>>& inputs)
    : m_type{Type::EXECUTE}
    , m_buffer_number{buffer_number}
    , m_data{nullptr}
    , m_size_in_bytes{0}
    , m_executable{executable}
    , m_outputs{&outputs}
    , m_inputs{&inputs}
{
    (void)m_buffer_number;
}

future<void>
    runtime::Backend::post_async_read_event(void* p, size_t size_in_bytes, size_t buffer_number)
{
    async_thread_start();
    auto event = make_shared<AsyncEvent>(AsyncEvent::Type::READ, buffer_number, p, size_in_bytes);
    unique_lock<std::mutex> lock(m_event_queue_mutex);
    m_event_queue.push_back(event);
    m_event_queue_condition.notify_one();
    return event->get_future();
}

future<void>
    runtime::Backend::post_async_write_event(void* p, size_t size_in_bytes, size_t buffer_number)
{
    async_thread_start();
    auto event = make_shared<AsyncEvent>(AsyncEvent::Type::WRITE, buffer_number, p, size_in_bytes);
    unique_lock<std::mutex> lock(m_event_queue_mutex);
    m_event_queue.push_back(event);
    m_event_queue_condition.notify_one();
    return event->get_future();
}

future<void> runtime::Backend::post_async_execute_event(
    const std::shared_ptr<Executable>& executable,
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs,
    size_t buffer_number)
{
    async_thread_start();
    auto event = make_shared<AsyncEvent>(buffer_number, executable, outputs, inputs);
    unique_lock<std::mutex> lock(m_event_queue_mutex);
    m_event_queue.push_back(event);
    m_event_queue_condition.notify_one();
    return event->get_future();
}

void runtime::Backend::async_thread_start()
{
    if (!m_event_queue_active)
    {
        m_event_queue_active = true;
        m_event_queue_thread =
            unique_ptr<thread>(new thread(&runtime::Backend::async_thread_entry, this));
    }
}

void runtime::Backend::async_thread_stop()
{
    if (m_event_queue_active)
    {
        {
            unique_lock<std::mutex> lk(m_event_queue_mutex);
            m_event_queue_active = false;
            m_event_queue_condition.notify_one();
        }
        m_event_queue_thread->join();
    }
}

void runtime::Backend::async_thread_process(const shared_ptr<AsyncEvent>& event)
{
    NGRAPH_INFO << "process";
}

void runtime::Backend::async_thread_entry()
{
    NGRAPH_INFO << "******************** inside thread";
    unique_lock<std::mutex> lk(m_event_queue_mutex);
    while (m_event_queue_active)
    {
        m_event_queue_condition.wait(lk);
        while (!m_event_queue.empty())
        {
            async_thread_process(m_event_queue.front());
            m_event_queue.pop_front();
        }
    }
}
