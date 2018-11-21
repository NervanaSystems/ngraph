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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/runtime/performance_counter.hpp"

/// performance test utilities
std::multimap<size_t, std::string>
    aggregate_timing(const std::vector<ngraph::runtime::PerformanceCounter>& perf_data);

std::vector<ngraph::runtime::PerformanceCounter> run_benchmark(std::shared_ptr<ngraph::Function> f,
                                                               const std::string& backend_name,
                                                               size_t iterations,
                                                               bool timing_detail,
                                                               int warmup_iterations,
                                                               bool copy_data);
