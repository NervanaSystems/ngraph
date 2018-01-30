// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
//
#include "ngraph/file_util.hpp"
#include "ngraph/json.hpp"
#include "ngraph/pass/simplification.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(simplification, remove_reshape)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Simplification>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/bn_fprop.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    size_t count_before = count_ops_of_type<op::Reshape>(func);
    pass_manager.run_passes(func);
    size_t count_after = count_ops_of_type<op::Reshape>(func);
}

TEST(simplification, remove_tranpose)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Simplification>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/tranpose.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    size_t count_before = count_ops_of_type<op::Reshape>(func);
    pass_manager.run_passes(func);
    size_t count_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_TRUE(count_after < count_before);
}

TEST(simplification, bn_bprop_rewrite)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Simplification>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/bn_bprop.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    size_t count_before = count_ops_of_type<op::Reshape>(func);
    pass_manager.run_passes(func);
    size_t count_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_TRUE(count_after < count_before);
}