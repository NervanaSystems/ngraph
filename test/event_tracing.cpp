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

#include <fstream>
#include <iostream>
#include <mutex>
#include <stdlib.h>
#include <vector>
#include "nlohmann/json.hpp"

#include "gtest/gtest.h"
#include "ngraph/event_tracing.hpp"
#include "ngraph/file_util.hpp"

using namespace std;

TEST(event_tracing, event_file)
{
    ngraph::Event::enable_event_tracing();
    std::vector<std::thread> threads;
    for (auto i = 0; i < 10; i++)
    {
        int id = i;
        std::thread next_thread([&] {
            std::ostringstream oss;
            oss << "Event: " << id;
            ngraph::Event event(oss.str(), "Dummy", "none");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            event.Stop();
            ngraph::Event::write_trace(event);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        threads.push_back(std::move(next_thread));
    }

    for (auto& next : threads)
    {
        next.join();
    }

    // Now read the file
    auto json_string = ngraph::file_util::read_file_to_string("ngraph_event_trace.json");
    nlohmann::json json_from_file(json_string);

    // Validate the JSON objects - there should be 10 of them
    // TODO
    ngraph::Event::disable_event_tracing();
}

TEST(event_tracing, event_writer_callback)
{
    // Create the event writer
    vector<ngraph::Event> event_list;
    auto event_writer = [&](const ngraph::Event& event) { event_list.push_back(event); };

    map<string, unique_ptr<ngraph::Event>> expected_event_table;
    mutex expected_event_table_mtx;

    ngraph::Event::enable_event_tracing();
    ngraph::Event::register_event_writer(event_writer);

    auto worker = [&](int worker_id) {
        std::ostringstream oss;
        oss << "Event: " << worker_id;
        unique_ptr<ngraph::Event> event(new ngraph::Event(oss.str(), "Dummy", "none"));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        event->Stop();
        ngraph::Event::write_trace(*event);

        lock_guard<mutex> lock(expected_event_table_mtx);
        expected_event_table[event->get_name()] = move(event);
    };

    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++)
    {
        std::thread thread_next(worker, i);
        threads.push_back(move(thread_next));
    }

    for (auto& next : threads)
    {
        next.join();
    }
    ngraph::Event::disable_event_tracing();

    // Now validate the events
    ASSERT_EQ(10, event_list.size());
    ASSERT_EQ(10, expected_event_table.size());

    for (const auto& next_event : event_list)
    {
        const auto& expected_event_key = expected_event_table.find(next_event.get_name());
        EXPECT_TRUE(expected_event_key != expected_event_table.end());
        EXPECT_EQ(expected_event_key->second->get_name(), next_event.get_name());
        EXPECT_EQ(expected_event_key->second->get_start(), next_event.get_start());
        EXPECT_EQ(expected_event_key->second->get_stop(), next_event.get_stop());
    }
}
