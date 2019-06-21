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
    // Set the environment variable to ensure logging
    ngraph::Event::enable_event_tracing();
    std::vector<std::thread> threads;
    std::mutex mtx;
    for (auto i = 0; i < 10; i++)
    {
        int id = i;
        std::thread next_thread([&] {
            std::ostringstream oss;
            oss << "Event: " << id;
            ngraph::Event event(oss.str(), "Dummy", "none");
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            event.Stop();
            ngraph::Event::write_trace(event);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        threads.push_back(std::move(next_thread));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

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
