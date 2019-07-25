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

#pragma once

#include <string>
#include <unordered_set>

namespace ngraph
{
    class BackendTest
    {
    public:
        static void set_backend_under_test(const std::string& backend_name);
        static const std::string& get_backend_under_test();
        static void load_manifest(const std::string& manifest_filename);

    private:
        static std::string s_backend_under_test;
    };
}
