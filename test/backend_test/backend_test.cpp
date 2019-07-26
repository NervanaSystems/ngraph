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
unordered_set<string> BackendTest::s_blacklist;
vector<TestPreRegistration*> BackendTest::s_pre_registrations;
bool BackendTest::s_registrations_finalized = false;

void BackendTest::set_backend_under_test(const string& backend_name)
{
    s_backend_under_test = backend_name;
}

const string& BackendTest::get_backend_under_test()
{
    return s_backend_under_test;
}

void BackendTest::load_manifest(const string& manifest_filename)
{
    ifstream f(manifest_filename);
    NGRAPH_CHECK(f.is_open(), "Could not open manifest file: ", manifest_filename);
    string line;
    while (getline(f, line))
    {
        size_t pound_pos = line.find('#');
        line = (pound_pos > line.size()) ? line : line.substr(0, pound_pos);
        line = trim(line);
        if (line.size() > 1)
        {
            s_blacklist.insert(line);
        }
    }
}

string BackendTest::prepend_disabled(const string& test_case_name, const string& test_name)
{
    string rc = test_name;
    string compound_test_name = test_case_name + "." + test_name;
    if (s_blacklist.find(test_name) != s_blacklist.end() ||
        s_blacklist.find(compound_test_name) != s_blacklist.end())
    {
        rc = "DISABLED_" + test_name;
    }
    return rc;
}

TestPreRegistration*
    BackendTest::pre_register_test(const char* test_case_name,
                                   const char* name,
                                   const char* type_param,
                                   const char* value_param,
                                   ::testing::internal::CodeLocation code_location,
                                   ::testing::internal::TypeId fixture_class_id,
                                   ::testing::internal::SetUpTestCaseFunc set_up_tc,
                                   ::testing::internal::TearDownTestCaseFunc tear_down_tc,
                                   ::testing::internal::TestFactoryBase* factory)
{
    TestPreRegistration* rc = new TestPreRegistration(test_case_name,
                                                      name,
                                                      type_param,
                                                      value_param,
                                                      code_location,
                                                      fixture_class_id,
                                                      set_up_tc,
                                                      tear_down_tc,
                                                      factory);
    s_pre_registrations.push_back(rc);
    return rc;
}

void BackendTest::finalize_test_registrations()
{
    NGRAPH_CHECK(!s_registrations_finalized,
                 "BackendTest::finalize_test_registrations() can only be called once");

    for (TestPreRegistration* pre_registration : s_pre_registrations)
    {
        pre_registration->m_test_info = ::testing::internal::MakeAndRegisterTestInfo(
            pre_registration->m_test_case_name,
            prepend_disabled(pre_registration->m_test_case_name, pre_registration->m_name).c_str(),
            nullptr,
            nullptr,
            pre_registration->m_code_location,
            pre_registration->m_fixture_class_id,
            pre_registration->m_set_up_tc,
            pre_registration->m_tear_down_tc,
            pre_registration->m_factory);
    }

    s_registrations_finalized = true;
}
