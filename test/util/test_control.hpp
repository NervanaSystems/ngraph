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
    std::string prepend_disabled(const std::string& test_case_name,
                                 const std::string& test_name,
                                 const std::string& manifest);
}

#define NGRAPH_GTEST_TEST_(test_case_name, test_name, parent_class, parent_id)                     \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name)                                        \
        : public parent_class                                                                      \
    {                                                                                              \
    public:                                                                                        \
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}                                     \
    private:                                                                                       \
        virtual void TestBody();                                                                   \
        static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;                      \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_case_name, test_name));        \
    };                                                                                             \
                                                                                                   \
    ::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::test_info_ =     \
        ::testing::internal::MakeAndRegisterTestInfo(                                              \
            #test_case_name,                                                                       \
            ::ngraph::prepend_disabled(#test_case_name, #test_name, s_manifest).c_str(),           \
            nullptr,                                                                               \
            nullptr,                                                                               \
            ::testing::internal::CodeLocation(__FILE__, __LINE__),                                 \
            (parent_id),                                                                           \
            parent_class::SetUpTestCase,                                                           \
            parent_class::TearDownTestCase,                                                        \
            new ::testing::internal::TestFactoryImpl<GTEST_TEST_CLASS_NAME_(test_case_name,        \
                                                                            test_name)>);          \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()

#define NGRAPH_TEST(test_case_name, test_name)                                                     \
    NGRAPH_GTEST_TEST_(                                                                            \
        test_case_name, test_name, ::testing::Test, ::testing::internal::GetTestTypeId())
