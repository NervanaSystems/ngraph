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

#include "backend_test/backend_test.hpp"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

string BackendTest::s_backend_under_test;

void BackendTest::set_backend_under_test(const string& backend_name)
{
    s_backend_under_test = backend_name;
}

const string& BackendTest::get_backend_under_test()
{
    return s_backend_under_test;
}

// TODO(amprocte): stub
void BackendTest::load_manifest(const string& manifest_filename)
{
}
