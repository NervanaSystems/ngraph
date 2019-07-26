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

#include "gtest/gtest.h"

namespace ngraph
{
    struct TestPreRegistration
    {
    public:
        TestPreRegistration(const char* test_case_name,
                            const char* name,
                            const char* type_param,
                            const char* value_param,
                            ::testing::internal::CodeLocation code_location,
                            ::testing::internal::TypeId fixture_class_id,
                            ::testing::internal::SetUpTestCaseFunc set_up_tc,
                            ::testing::internal::TearDownTestCaseFunc tear_down_tc,
                            ::testing::internal::TestFactoryBase* factory)
            : m_test_case_name(test_case_name)
            , m_name(name)
            , m_type_param(type_param)
            , m_value_param(value_param)
            , m_code_location(code_location)
            , m_fixture_class_id(fixture_class_id)
            , m_set_up_tc(set_up_tc)
            , m_tear_down_tc(tear_down_tc)
            , m_factory(factory)
            , m_test_info(nullptr)
        {
        }
        const char* m_test_case_name;
        const char* m_name;
        const char* m_type_param;
        const char* m_value_param;
        ::testing::internal::CodeLocation m_code_location;
        ::testing::internal::TypeId m_fixture_class_id;
        ::testing::internal::SetUpTestCaseFunc m_set_up_tc;
        ::testing::internal::TearDownTestCaseFunc m_tear_down_tc;
        ::testing::internal::TestFactoryBase* m_factory;
        ::testing::TestInfo* m_test_info;
    };

    class BackendTest
    {
    public:
        static void set_backend_under_test(const std::string& backend_name);
        static const std::string& get_backend_under_test();
        static void load_manifest(const std::string& manifest_filename);
        static TestPreRegistration*
            pre_register_test(const char* test_case_name,
                              const char* name,
                              const char* type_param,
                              const char* value_param,
                              ::testing::internal::CodeLocation code_location,
                              ::testing::internal::TypeId fixture_class_id,
                              ::testing::internal::SetUpTestCaseFunc set_up_tc,
                              ::testing::internal::TearDownTestCaseFunc tear_down_tc,
                              ::testing::internal::TestFactoryBase* factory);
        static void finalize_test_registrations();

    private:
        static std::string prepend_disabled(const std::string& test_case_name,
                                            const std::string& test_name);
        static std::string s_backend_under_test;
        static std::unordered_set<std::string> s_blacklist;
        static std::vector<TestPreRegistration*> s_pre_registrations;
        static bool s_registrations_finalized;
    };
}

#define BACKEND_GTEST_TEST_CLASS_NAME_(test_case_name, test_name)                                  \
    test_case_name##_##test_name##_Test

#define BACKEND_GTEST_TEST_(test_case_name, test_name, parent_class, parent_id)                    \
    class BACKEND_GTEST_TEST_CLASS_NAME_(test_case_name, test_name)                                \
        : public parent_class                                                                      \
    {                                                                                              \
    public:                                                                                        \
        BACKEND_GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}                             \
    private:                                                                                       \
        virtual void TestBody();                                                                   \
        static ::ngraph::TestPreRegistration* const test_pre_registration_                         \
            GTEST_ATTRIBUTE_UNUSED_;                                                               \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(BACKEND_GTEST_TEST_CLASS_NAME_(test_case_name,             \
                                                                       test_name));                \
    };                                                                                             \
                                                                                                   \
    ::ngraph::TestPreRegistration* const BACKEND_GTEST_TEST_CLASS_NAME_(                           \
        test_case_name, test_name)::test_pre_registration_ =                                       \
        ::ngraph::BackendTest::pre_register_test(                                                  \
            #test_case_name,                                                                       \
            #test_name,                                                                            \
            nullptr,                                                                               \
            nullptr,                                                                               \
            ::testing::internal::CodeLocation(__FILE__, __LINE__),                                 \
            (parent_id),                                                                           \
            parent_class::SetUpTestCase,                                                           \
            parent_class::TearDownTestCase,                                                        \
            new ::testing::internal::TestFactoryImpl<BACKEND_GTEST_TEST_CLASS_NAME_(               \
                test_case_name, test_name)>);                                                      \
    void BACKEND_GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()

#define BACKEND_TEST(test_case_name, test_name)                                                    \
    BACKEND_GTEST_TEST_(                                                                           \
        test_case_name, test_name, ::testing::Test, ::testing::internal::GetTestTypeId())
