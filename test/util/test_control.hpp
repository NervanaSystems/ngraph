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

#define NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)                     \
    backend_name##_##test_case_name##_##test_name##_Test

#define NGRAPH_GTEST_TEST_(backend_name, test_case_name, test_name, parent_class, parent_id)       \
    class NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)                   \
        : public parent_class                                                                      \
    {                                                                                              \
    public:                                                                                        \
        NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)() {}                \
    private:                                                                                       \
        virtual void TestBody();                                                                   \
        static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;                      \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name,                \
                                                                      test_case_name,              \
                                                                      test_name));                 \
    };                                                                                             \
                                                                                                   \
    ::testing::TestInfo* const NGRAPH_GTEST_TEST_CLASS_NAME_(                                      \
        backend_name, test_case_name, test_name)::test_info_ =                                     \
        ::testing::internal::MakeAndRegisterTestInfo(                                              \
            ::ngraph::combine_test_backend_and_case(#backend_name, #test_case_name).c_str(),       \
            ::ngraph::prepend_disabled(#backend_name, #test_name, s_manifest).c_str(),             \
            nullptr,                                                                               \
            nullptr,                                                                               \
            ::testing::internal::CodeLocation(__FILE__, __LINE__),                                 \
            (parent_id),                                                                           \
            parent_class::SetUpTestCase,                                                           \
            parent_class::TearDownTestCase,                                                        \
            new ::testing::internal::TestFactoryImpl<NGRAPH_GTEST_TEST_CLASS_NAME_(                \
                backend_name, test_case_name, test_name)>);                                        \
    void NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)::TestBody()

#define NGRAPH_TEST(test_case_name, test_name)                                                     \
    NGRAPH_GTEST_TEST_(test_case_name,                                                             \
                       test_case_name,                                                             \
                       test_name,                                                                  \
                       ::testing::Test,                                                            \
                       ::testing::internal::GetTestTypeId())

// NGRAPH_TEST_F facilitates the use of the same configuration parameters for multiple
// unit tests similar to the original TEST_F, but with the introduction of a new 0th
// parameter for the backend name, which allows nGraph's manifest controlled unit testing.
//
// Start by defining a class derived from testing::Test, which you'll pass for the
// text_fixture parameter.
// Then use this class to define multiple related unit tests (which share some common
// configuration information and/or setup code).
//
// Generated test names take the form:
// BACKENDNAME/FixtureClassName.test_name
// where the test case name is "BACKENDNAME/FixtureClassName"
// and the test name is "test_name"
//
// With the use of NGRAPH_TEST_F the filter to run all the tests for a given backend
// should be:
// --gtest_filter=BACKENDNAME*.*
// (rather than the BACKENDNAME.* that worked before the use of NGRAPH_TEST_F)
#define NGRAPH_TEST_F(backend_name, test_fixture, test_name)                                       \
    NGRAPH_GTEST_TEST_(backend_name,                                                               \
                       test_fixture,                                                               \
                       test_name,                                                                  \
                       test_fixture,                                                               \
                       ::testing::internal::GetTypeId<test_fixture>())

// NGRAPH_TEST_P combined with NGRAPH_INSTANTIATE_TEST_CASE_P facilate the generation
// of value parameterized tests (similar to the original TEST_P and INSTANTIATE_TEST_CASE_P
// with the addition of a new 0th parameter for the backend name, which allows nGraph's
// manifest controlled unit testing).
//
// Start by defining a class derived from ::testing::TestWithParam<T>, which you'll pass
// for the test_case_name parameter.
// Then use NGRAPH_INSTANTIATE_TEST_CASE_P to define each generation of test cases (see below).
#define NGRAPH_TEST_P(backend_name, test_case_name, test_name)                                     \
    class NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)                   \
        : public test_case_name                                                                    \
    {                                                                                              \
    public:                                                                                        \
        NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)() {}                \
        virtual void TestBody();                                                                   \
                                                                                                   \
    private:                                                                                       \
        static int AddToRegistry()                                                                 \
        {                                                                                          \
            ::testing::UnitTest::GetInstance()                                                     \
                ->parameterized_test_registry()                                                    \
                .GetTestCasePatternHolder<test_case_name>(                                         \
                    #backend_name "/" #test_case_name,                                             \
                    ::testing::internal::CodeLocation(__FILE__, __LINE__))                         \
                ->AddTestPattern(                                                                  \
                    #backend_name "/" #test_case_name,                                             \
                    ::ngraph::prepend_disabled(#test_case_name, #test_name, s_manifest).c_str(),   \
                    new ::testing::internal::TestMetaFactory<NGRAPH_GTEST_TEST_CLASS_NAME_(        \
                        backend_name, test_case_name, test_name)>());                              \
            return 0;                                                                              \
        }                                                                                          \
        static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;                               \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name,                \
                                                                      test_case_name,              \
                                                                      test_name));                 \
    };                                                                                             \
    int NGRAPH_GTEST_TEST_CLASS_NAME_(                                                             \
        backend_name, test_case_name, test_name)::gtest_registering_dummy_ =                       \
        NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)::AddToRegistry();   \
    void NGRAPH_GTEST_TEST_CLASS_NAME_(backend_name, test_case_name, test_name)::TestBody()

// Use NGRAPH_INSTANTIATE_TEST_CASE_P to create a generated set of test case variations.
// The prefix parameter is a label that you optionally provide (no quotes) for a unique
// test name (helpful for labelling a set of inputs and for filtering).
// The prefix parameter can be skipped by simply using a bare comma (see example below).
//
// Unlike INSTANTIATE_TEST_CASE_P we don't currently support passing a custom param
// name generator. Supporting this with a single macro name requires the use of
// ... and __VA_ARGS__ which in turn generates a warning using INSTANTIATE_TEST_CASE_P
// without a trailing , parameter.
//
// Examples:
// NGRAPH_INSTANTIATE_TEST_CASE_P(BACKENDNAME,                  // backend_name
//                                ,                             // empty/skipped prefix
//                                TestWithParamSubClass,        // test_case_name
//                                ::testing::Range(0, 3) )      // test generator
// would generate:
// BACKENDNAME/TestWithParamSubClass.test_name/0
// BACKENDNAME/TestWithParamSubClass.test_name/1
// BACKENDNAME/TestWithParamSubClass.test_name/2
//
// NGRAPH_INSTANTIATE_TEST_CASE_P(BACKENDNAME,                  // backend_name
//                                NumericRangeTests,            // prefix
//                                TestWithParamSubClass,        // test_case_name
//                                ::testing::Range(0, 3) )      // test generator
// would generate:
// BACKENDNAME/NumericRangeTests/BACKENDNAME/TestWithParamSubClass.test_name/0
// BACKENDNAME/NumericRangeTests/BACKENDNAME/TestWithParamSubClass.test_name/1
// BACKENDNAME/NumericRangeTests/BACKENDNAME/TestWithParamSubClass.test_name/2
//
// With the use of NGRAPH_TEST_P and NGRAPH_INSTANTIATE_TEST_CASE_P
// the filter to run all the tests for a given backend should be:
// --gtest_filter=BACKENDNAME*.*
// (rather than the BACKENDNAME.* that worked before the use of NGRAPH_TEST_P)
#define NGRAPH_INSTANTIATE_TEST_CASE_P(backend_name, prefix, test_case_name, generator)            \
    static ::testing::internal::ParamGenerator<test_case_name::ParamType>                          \
        gtest_##prefix##backend_name##test_case_name##_EvalGenerator_()                            \
    {                                                                                              \
        return generator;                                                                          \
    }                                                                                              \
    static ::std::string gtest_##prefix##backend_name##test_case_name##_EvalGenerateName_(         \
        const ::testing::TestParamInfo<test_case_name::ParamType>& info)                           \
    {                                                                                              \
        return ::testing::internal::GetParamNameGen<test_case_name::ParamType>()(info);            \
    }                                                                                              \
    static int gtest_##prefix##backend_name##test_case_name##_dummy_ GTEST_ATTRIBUTE_UNUSED_ =     \
        ::testing::UnitTest::GetInstance()                                                         \
            ->parameterized_test_registry()                                                        \
            .GetTestCasePatternHolder<test_case_name>(                                             \
                #backend_name "/" #test_case_name,                                                 \
                ::testing::internal::CodeLocation(__FILE__, __LINE__))                             \
            ->AddTestCaseInstantiation(                                                            \
                #prefix[0] != '\0' ? #backend_name "/" #prefix : "",                               \
                &gtest_##prefix##backend_name##test_case_name##_EvalGenerator_,                    \
                &gtest_##prefix##backend_name##test_case_name##_EvalGenerateName_,                 \
                __FILE__,                                                                          \
                __LINE__)
