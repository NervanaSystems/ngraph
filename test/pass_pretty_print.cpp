/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pretty_print.hpp"

#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(pretty_print, existing_models)
{
    vector<string> models = {"mxnet/mnist_mlp_forward.json",
                             "mxnet/10_bucket_LSTM.json",
                             "mxnet/LSTM_backward.json",
                             "mxnet/LSTM_forward.json"};

    for (const string& model : models)
    {
        const string json_path = file_util::path_join(SERIALIZED_ZOO, model);
        const string json_string = file_util::read_file_to_string(json_path);
        shared_ptr<Function> f = ngraph::deserialize(json_string);

        pass::Manager pass_manager;
        ofstream file(json_path + ".pretty");
        pass_manager.register_pass<pass::PrettyPrint>(file);
        pass_manager.run_passes(f);
        file.close();
    }
}
