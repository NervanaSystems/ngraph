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

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <stdio.h>
#include <vector>

#include "gtest/gtest.h"
#include "misc.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_debug_tracer.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

static void set_env_vars(const string& trace_log, const string& bin_log)
{
    set_environment("NGRAPH_CPU_DEBUG_TRACER", "1", 1);
    set_environment("NGRAPH_CPU_TRACER_LOG", trace_log.c_str(), 1);
    set_environment("NGRAPH_CPU_BIN_TRACER_LOG", bin_log.c_str(), 1);
}

static void unset_env_vars()
{
    unset_environment("NGRAPH_CPU_DEBUG_TRACER");
    unset_environment("NGRAPH_CPU_TRACER_LOG");
    unset_environment("NGRAPH_CPU_BIN_TRACER_LOG");
}

static void open_logs(ifstream& meta, ifstream& bin, const string& trace_log, const string& bin_log)
{
    meta.open(trace_log);
    bin.open(bin_log, std::ios::binary);

    ASSERT_TRUE(meta.is_open());
    ASSERT_TRUE(bin.is_open());
}

TEST(cpu_debug_tracer, MLIR_DISABLE_TEST(check_flow_with_external_function))
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, vector<float>{0, 1, 2, 3});
    copy_data(b, vector<float>{1, 2, 3, 4});

    const string trace_log_file = "trace_meta.log";
    const string bin_log_file = "trace_bin_data.log";

    set_env_vars(trace_log_file, bin_log_file);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    // open two logs and parse them
    ifstream f_meta;
    ifstream f_bin;
    open_logs(f_meta, f_bin, trace_log_file, bin_log_file);

    string line;

    getline(f_meta, line);
    auto str_mean = line.substr(line.find("mean"));
    auto mean =
        std::stod(str_mean.substr(str_mean.find("=") + 1, str_mean.find(" ") - str_mean.find("=")));

    // mean value of first tensor - a
    EXPECT_EQ(mean, 1.5);

    getline(f_meta, line);
    auto str_var = line.substr(line.find("var"));
    auto var = std::stod(str_var.substr(str_var.find("=") + 1));

    // variance value of second tensor - b
    EXPECT_EQ(var, 1.25);

    getline(f_meta, line);
    auto str_bin_offset = line.substr(line.find("bin_data"));
    auto bin_offset = std::stod(str_bin_offset.substr(
        str_bin_offset.find("=") + 1, str_bin_offset.find(" ") - str_bin_offset.find("=")));

    // check output tensor from binary data
    f_bin.seekg(bin_offset);

    std::vector<unsigned char> v_c((std::istreambuf_iterator<char>(f_bin)),
                                   std::istreambuf_iterator<char>());

    vector<float> v_f(4);
    memcpy(&v_f[0], &v_c[0], sizeof(float) * 4);

    EXPECT_EQ((vector<float>{1, 3, 5, 7}), (v_f));

    remove(trace_log_file.c_str());
    remove(bin_log_file.c_str());
    unset_env_vars();
}
