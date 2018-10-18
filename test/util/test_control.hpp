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

#include <string>

#include "gtest/gtest.h"

// Copied from gtest

namespace ngraph
{
    std::string prepend_disabled(const std::string& backend_name,
                                 const std::string& test_name,
                                 const std::string& manifest);

    std::string combine_test_backend_and_case(const std::string& backend_name,
                                              const std::string& test_casename);
}

#define NGRAPH_GTEST_TEST_(backend_name, test_case_name, test_name, parent_class, parent_id)       \
    class GTEST_TEST_CLASS_NAME_(backend_name, test_name)                                          \
        : public parent_class                                                                      \
    {                                                                                              \
    public:                                                                                        \
        GTEST_TEST_CLASS_NAME_(backend_name, test_name)() {}                                       \
    private:                                                                                       \
        virtual void TestBody();                                                                   \
        static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;                      \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(backend_name, test_name));          \
    };                                                                                             \
                                                                                                   \
    ::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(backend_name, test_name)::test_info_ =       \
        ::testing::internal::MakeAndRegisterTestInfo(                                              \
            ::ngraph::combine_test_backend_and_case(#backend_name, #test_case_name).c_str(),       \
            ::ngraph::prepend_disabled(#backend_name, #test_name, s_manifest).c_str(),             \
            nullptr,                                                                               \
            nullptr,                                                                               \
            ::testing::internal::CodeLocation(__FILE__, __LINE__),                                 \
            (parent_id),                                                                           \
            parent_class::SetUpTestCase,                                                           \
            parent_class::TearDownTestCase,                                                        \
            new ::testing::internal::TestFactoryImpl<GTEST_TEST_CLASS_NAME_(backend_name,          \
                                                                            test_name)>);          \
    void GTEST_TEST_CLASS_NAME_(backend_name, test_name)::TestBody()

#define NGRAPH_TEST(test_case_name, test_name)                                                     \
    NGRAPH_GTEST_TEST_(test_case_name,                                                             \
                       test_case_name,                                                             \
                       test_name,                                                                  \
                       ::testing::Test,                                                            \
                       ::testing::internal::GetTestTypeId())

#define NGRAPH_TEST_F(backend_name, test_fixture, test_name)                                       \
    NGRAPH_GTEST_TEST_(backend_name,                                                               \
                       test_fixture,                                                               \
                       test_name,                                                                  \
                       test_fixture,                                                               \
                       ::testing::internal::GetTypeId<test_fixture>())

#define NGRAPH_TEST_P(backend_name, test_case_name, test_name)                                     \
    class GTEST_TEST_CLASS_NAME_(backend_name, test_name)                                          \
        : public test_case_name                                                                    \
    {                                                                                              \
    public:                                                                                        \
        GTEST_TEST_CLASS_NAME_(backend_name, test_name)() {}                                       \
        virtual void TestBody();                                                                   \
                                                                                                   \
    private:                                                                                       \
        static int AddToRegistry()                                                                 \
        {                                                                                          \
            ::testing::UnitTest::GetInstance()                                                     \
                ->parameterized_test_registry()                                                    \
                .GetTestCasePatternHolder<test_case_name>(                                         \
                    #backend_name "_" #test_case_name,                                             \
                    ::testing::internal::CodeLocation(__FILE__, __LINE__))                         \
                ->AddTestPattern(                                                                  \
                    #backend_name "_" #test_case_name,                                             \
                    ::ngraph::prepend_disabled(#test_case_name, #test_name, s_manifest).c_str(),   \
                    new ::testing::internal::TestMetaFactory<GTEST_TEST_CLASS_NAME_(               \
                        backend_name, test_name)>());                                              \
            return 0;                                                                              \
        }                                                                                          \
        static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;                               \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(backend_name, test_name));          \
    };                                                                                             \
    int GTEST_TEST_CLASS_NAME_(backend_name, test_name)::gtest_registering_dummy_ =                \
        GTEST_TEST_CLASS_NAME_(backend_name, test_name)::AddToRegistry();                          \
    void GTEST_TEST_CLASS_NAME_(backend_name, test_name)::TestBody()

#define NGRAPH_INSTANTIATE_TEST_CASE_P(backend_name, prefix, test_case_name, generator, ...)       \
    ::testing::internal::ParamGenerator<test_case_name::ParamType>                                 \
        gtest_##prefix##backend_name##test_case_name##_EvalGenerator_()                            \
    {                                                                                              \
        return generator;                                                                          \
    }                                                                                              \
    ::std::string gtest_##prefix##backend_name##test_case_name##_EvalGenerateName_(                \
        const ::testing::TestParamInfo<test_case_name::ParamType>& info)                           \
    {                                                                                              \
        return ::testing::internal::GetParamNameGen<test_case_name::ParamType>(__VA_ARGS__)(info); \
    }                                                                                              \
    static int gtest_##prefix##backend_name##test_case_name##_dummy_ GTEST_ATTRIBUTE_UNUSED_ =     \
        ::testing::UnitTest::GetInstance()                                                         \
            ->parameterized_test_registry()                                                        \
            .GetTestCasePatternHolder<test_case_name>(                                             \
                #backend_name "_" #test_case_name,                                                 \
                ::testing::internal::CodeLocation(__FILE__, __LINE__))                             \
            ->AddTestCaseInstantiation(                                                            \
                #backend_name "_" #prefix,                                                         \
                &gtest_##prefix##backend_name##test_case_name##_EvalGenerator_,                    \
                &gtest_##prefix##backend_name##test_case_name##_EvalGenerateName_,                 \
                __FILE__,                                                                          \
                __LINE__)
