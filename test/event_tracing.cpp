//*****************************************************************************
// Copyright 2019 Intel Corporation
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
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

TEST(event_tracing, duration)
{
    event::Manager::enable_event_tracing();
    vector<thread> threads;
    mutex mtx;
    for (auto i = 0; i < 10; i++)
    {
        int id = i;
        thread next_thread([&] {
            ostringstream oss;
            oss << "Event: " << id;
            event::Duration event(oss.str(), "Dummy");
            this_thread::sleep_for(chrono::milliseconds(2));
            event.stop();
        });
        this_thread::sleep_for(chrono::milliseconds(2));
        threads.push_back(move(next_thread));
    }

    this_thread::sleep_for(chrono::milliseconds(200));

    for (auto& next : threads)
    {
        next.join();
    }

    event::Manager::close();

    // Now read the file
    auto json_string = ngraph::file_util::read_file_to_string("ngraph_event_trace.json");
    nlohmann::json json_from_file = nlohmann::json::parse(json_string);

    EXPECT_EQ(10, json_from_file.size());

    event::Manager::disable_event_tracing();
}

TEST(event_tracing, object)
{
    event::Manager::enable_event_tracing();

    vector<event::Object> objects;
    for (size_t i = 0; i < 10; ++i)
    {
        stringstream ss;
        ss << "object_" << i;
        nlohmann::json args;
        args["arg0"] = i * 10;
        args["arg1"] = i * 20;
        args["arg2"] = i * 30;
        objects.emplace_back(ss.str(), args);
    }
    this_thread::sleep_for(chrono::milliseconds(10));
    for (event::Object& obj : objects)
    {
        nlohmann::json args;
        args["arg0.1"] = "one";
        args["arg1.1"] = "two";
        args["arg2.1"] = "three";
        obj.snapshot(args);
    }
    this_thread::sleep_for(chrono::milliseconds(10));
    for (event::Object& obj : objects)
    {
        obj.destroy();
    }

    event::Manager::close();

    event::Manager::disable_event_tracing();
}

TEST(benchmark, event_tracing)
{
    size_t outer_size = 10000;
    size_t inner_size = 100;

    event::Manager::enable_event_tracing();
    {
        ngraph::stopwatch timer;
        timer.start();
        for (size_t outer = 0; outer < outer_size; ++outer)
        {
            event::Duration outer_event("outer", "Dummy");
            for (size_t inner = 0; inner < inner_size; ++inner)
            {
                event::Duration inner_event("inner", "Dummy");
                inner_event.stop();
            }
            outer_event.stop();
        }
        timer.stop();
        NGRAPH_INFO << "enabled time " << timer.get_milliseconds() << "ms";
        NGRAPH_INFO << "enabled time "
                    << static_cast<double>(timer.get_milliseconds()) /
                           (outer_size * inner_size + outer_size)
                    << "ms per call";
    }

    event::Manager::disable_event_tracing();
    {
        ngraph::stopwatch timer;
        timer.start();
        for (size_t outer = 0; outer < outer_size; ++outer)
        {
            event::Duration outer_event("outer", "Dummy");
            for (size_t inner = 0; inner < inner_size; ++inner)
            {
                event::Duration inner_event("inner", "Dummy");
                inner_event.stop();
            }
            outer_event.stop();
        }
        timer.stop();
        NGRAPH_INFO << "disabled time " << timer.get_milliseconds() << "ms";
        NGRAPH_INFO << "disabled time "
                    << static_cast<double>(timer.get_milliseconds()) /
                           (outer_size * inner_size + outer_size)
                    << "ms per call";
    }
}
