//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include <cstdlib>
#include <string>
#include <tuple>
#include "gtest/gtest.h"

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"

namespace ngraph
{
    enum backend
    {
        INTERPRETER,
        CPU,
        GPU,
        INTELGPU,
        PlaidML
    };

    static std::string s_manifest = "${MANIFEST}";

    template <backend BACKEND_TARGET, backend BACKEND_REFERENCE>
    class model_comparison : public ::testing::TestWithParam<std::string>
    {
    public:
        void compare_results(ngraph::NodeVector& result_nodes,
                             std::vector<std::shared_ptr<ngraph::runtime::Tensor>> ref_results,
                             std::vector<std::shared_ptr<ngraph::runtime::Tensor>> bk_results)
        {
            for (int i = 0; i < ref_results.size(); ++i)
            {
                const std::shared_ptr<ngraph::runtime::Tensor>& ref_data = ref_results.at(i);
                const std::shared_ptr<ngraph::runtime::Tensor>& bk_data = bk_results.at(i);

                std::cout << "Comparing results for " << result_nodes.at(i)->get_name()
                          << std::endl;
                if (auto node =
                        std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(result_nodes.at(i)))
                {
                    std::cout << "  Parent node: ";
                    for (auto& p : node->get_arguments())
                    {
                        std::cout << " " << p->get_name() << std::endl;
                        std::cout << "   nargs: " << p->get_arguments().size() << std::endl;
                    }
                }

                ASSERT_EQ(ref_data->get_element_type(), bk_data->get_element_type());
                ASSERT_EQ(ref_data->get_element_count(), bk_data->get_element_count());
                ASSERT_EQ(ref_data->get_shape(), bk_data->get_shape());

                element::Type et = ref_data->get_element_type();
                if (et == element::boolean)
                {
                    std::vector<char> ref_data_vector = read_vector<char>(ref_data);
                    std::vector<char> bk_data_vector = read_vector<char>(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close<char>(ref_data_vector, bk_data_vector));
                }
                else if ((et == element::f32) || (et == element::f64))
                {
                    std::vector<float> ref_data_vector = read_float_vector(ref_data);
                    std::vector<float> bk_data_vector = read_float_vector(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close_f(ref_data_vector, bk_data_vector));
                }
                else if (et == element::i8)
                {
                    std::vector<int8_t> ref_data_vector = read_vector<int8_t>(ref_data);
                    std::vector<int8_t> bk_data_vector = read_vector<int8_t>(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close<int8_t>(ref_data_vector, bk_data_vector));
                }
                else if (et == element::i16)
                {
                    std::vector<int16_t> ref_data_vector = read_vector<int16_t>(ref_data);
                    std::vector<int16_t> bk_data_vector = read_vector<int16_t>(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close<int16_t>(ref_data_vector, bk_data_vector));
                }
                else if (et == element::i32)
                {
                    std::vector<int32_t> ref_data_vector = read_vector<int32_t>(ref_data);
                    std::vector<int32_t> bk_data_vector = read_vector<int32_t>(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close<int32_t>(ref_data_vector, bk_data_vector));
                }
                else if (et == element::i64)
                {
                    std::vector<int64_t> ref_data_vector = read_vector<int64_t>(ref_data);
                    std::vector<int64_t> bk_data_vector = read_vector<int64_t>(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close<int64_t>(ref_data_vector, bk_data_vector));
                }
                else if (et == element::u8)
                {
                    std::vector<uint8_t> ref_data_vector = read_vector<uint8_t>(ref_data);
                    std::vector<uint8_t> bk_data_vector = read_vector<uint8_t>(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close<uint8_t>(ref_data_vector, bk_data_vector));
                }
                else if (et == element::u16)
                {
                    std::vector<uint16_t> ref_data_vector = read_vector<uint16_t>(ref_data);
                    std::vector<uint16_t> bk_data_vector = read_vector<uint16_t>(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close<uint16_t>(ref_data_vector, bk_data_vector));
                }
                else if (et == element::u32)
                {
                    std::vector<uint32_t> ref_data_vector = read_vector<uint32_t>(ref_data);
                    std::vector<uint32_t> bk_data_vector = read_vector<uint32_t>(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close<uint32_t>(ref_data_vector, bk_data_vector));
                }
                else if (et == element::u64)
                {
                    std::vector<uint64_t> ref_data_vector = read_vector<uint64_t>(ref_data);
                    std::vector<uint64_t> bk_data_vector = read_vector<uint64_t>(bk_data);
                    print_results(ref_data_vector, bk_data_vector);
                    EXPECT_TRUE(test::all_close<uint64_t>(ref_data_vector, bk_data_vector));
                }
                else
                {
                    throw std::runtime_error("unsupported type");
                }
            }
        }

        std::pair<std::shared_ptr<ngraph::runtime::Backend>,
                  std::shared_ptr<ngraph::runtime::Backend>>
            get_backends()
        {
            std::shared_ptr<ngraph::runtime::Backend> b, r;
            switch (BACKEND_TARGET)
            {
            case backend::INTERPRETER: b = ngraph::runtime::Backend::create("INTERPRETER"); break;
            case backend::CPU: b = ngraph::runtime::Backend::create("CPU"); break;
            case backend::GPU: b = ngraph::runtime::Backend::create("GPU"); break;
            case backend::INTELGPU: b = ngraph::runtime::Backend::create("INTELGPU"); break;
            case backend::PlaidML: b = ngraph::runtime::Backend::create("PlaidML"); break;
            default: throw ngraph_error("Unregistered backend requested for graph comparison");
            }
            switch (BACKEND_REFERENCE)
            {
            case backend::INTERPRETER: r = ngraph::runtime::Backend::create("INTERPRETER"); break;
            case backend::CPU: r = ngraph::runtime::Backend::create("CPU"); break;
            case backend::GPU: r = ngraph::runtime::Backend::create("GPU"); break;
            case backend::INTELGPU: r = ngraph::runtime::Backend::create("INTELGPU"); break;
            case backend::PlaidML: r = ngraph::runtime::Backend::create("PlaidML"); break;
            default:
                throw ngraph_error("Unregistered reference backend requested for graph comparison");
            }
            return std::make_pair(b, r);
        }

    protected:
        model_comparison() { file_name = GetParam(); }
        std::string file_name;
    };
}

#define NGRAPH_COMPARISON_TEST_P(test_case_name)                                                   \
    class test_case_name##_Test : public test_case_name                                            \
    {                                                                                              \
    public:                                                                                        \
        test_case_name##_Test() {}                                                                 \
        virtual void TestBody();                                                                   \
                                                                                                   \
    private:                                                                                       \
        static int AddToRegistry()                                                                 \
        {                                                                                          \
            ::testing::UnitTest::GetInstance()                                                     \
                ->parameterized_test_registry()                                                    \
                .GetTestCasePatternHolder<test_case_name>(                                         \
                    #test_case_name, ::testing::internal::CodeLocation(__FILE__, __LINE__))        \
                ->AddTestPattern(                                                                  \
                    #test_case_name,                                                               \
                    "",                                                                            \
                    new ::testing::internal::TestMetaFactory<test_case_name##_Test>());            \
            return 0;                                                                              \
        }                                                                                          \
        static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;                               \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(test_case_name##_Test);                                    \
    };                                                                                             \
    int test_case_name##_Test::gtest_registering_dummy_ = test_case_name##_Test::AddToRegistry();  \
    void test_case_name##_Test::TestBody()

#define NGRAPH_COMPARISON_TEST(prefix, test_case_name, generator)                                  \
    NGRAPH_COMPARISON_TEST_P(test_case_name)                                                       \
    {                                                                                              \
        std::shared_ptr<ngraph::runtime::Backend> ref, backend;                                    \
        std::tie(backend, ref) = get_backends();                                                   \
        std::string frozen_graph_path;                                                             \
        if (file_name[0] != '/')                                                                   \
        {                                                                                          \
            frozen_graph_path = ngraph::file_util::path_join(SERIALIZED_ZOO, file_name);           \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            frozen_graph_path = file_name;                                                         \
        }                                                                                          \
        std::cout << frozen_graph_path << std::endl;                                               \
        std::stringstream ss(frozen_graph_path);                                                   \
        std::shared_ptr<ngraph::Function> func = ngraph::deserialize(ss);                          \
        ngraph::NodeVector new_results;                                                            \
        for (auto n : func->get_ordered_ops())                                                     \
        {                                                                                          \
            if (!n->is_output() && !n->is_parameter() && !n->is_constant() &&                      \
                !(n->get_outputs().size() > 1))                                                    \
            {                                                                                      \
                new_results.push_back(n);                                                          \
            }                                                                                      \
        }                                                                                          \
        auto new_func = std::make_shared<ngraph::Function>(new_results, func->get_parameters());   \
        auto ref_func = clone_function(*new_func);                                                 \
        auto bk_func = clone_function(*new_func);                                                  \
        std::vector<std::shared_ptr<ngraph::runtime::Tensor>> ref_args;                            \
        std::vector<std::shared_ptr<ngraph::runtime::Tensor>> bk_args;                             \
        std::default_random_engine engine(2112);                                                   \
        for (std::shared_ptr<ngraph::op::Parameter> param : new_func->get_parameters())            \
        {                                                                                          \
            auto data = std::make_shared<ngraph::runtime::HostTensor>(param->get_element_type(),   \
                                                                      param->get_shape());         \
            random_init(data.get(), engine);                                                       \
            auto ref_tensor = ref->create_tensor(param->get_element_type(), param->get_shape());   \
            auto bk_tensor =                                                                       \
                backend->create_tensor(param->get_element_type(), param->get_shape());             \
            ref_tensor->write(data->get_data_ptr(),                                                \
                              0,                                                                   \
                              data->get_element_count() * data->get_element_type().size());        \
            bk_tensor->write(data->get_data_ptr(),                                                 \
                             0,                                                                    \
                             data->get_element_count() * data->get_element_type().size());         \
            ref_args.push_back(ref_tensor);                                                        \
            bk_args.push_back(bk_tensor);                                                          \
        }                                                                                          \
        std::vector<std::shared_ptr<ngraph::runtime::Tensor>> ref_results;                         \
        std::vector<std::shared_ptr<ngraph::runtime::Tensor>> bk_results;                          \
        ref_results.reserve(new_results.size());                                                   \
        bk_results.reserve(new_results.size());                                                    \
        for (std::shared_ptr<ngraph::Node> & out : new_results)                                    \
        {                                                                                          \
            auto ref_result = ref->create_tensor(out->get_element_type(), out->get_shape());       \
            ref_results.push_back(ref_result);                                                     \
            auto bk_result = backend->create_tensor(out->get_element_type(), out->get_shape());    \
            bk_results.push_back(bk_result);                                                       \
        }                                                                                          \
        if (ref_func->get_parameters().size() != ref_args.size())                                  \
        {                                                                                          \
            throw ngraph::ngraph_error(                                                            \
                "Number of ref runtime parameters and allocated arguments don't match");           \
        }                                                                                          \
        if (bk_func->get_parameters().size() != bk_args.size())                                    \
        {                                                                                          \
            throw ngraph::ngraph_error(                                                            \
                "Number of backend runtime parameters and allocated arguments don't match");       \
        }                                                                                          \
        if (ref_func->get_results().size() != ref_results.size())                                  \
        {                                                                                          \
            throw ngraph::ngraph_error(                                                            \
                "Number of ref runtime results and allocated results don't match");                \
        }                                                                                          \
        if (bk_func->get_results().size() != bk_results.size())                                    \
        {                                                                                          \
            throw ngraph::ngraph_error(                                                            \
                "Number of backend runtime results and allocated results don't match");            \
        }                                                                                          \
        ref->call_with_validate(ref_func, ref_results, ref_args);                                  \
        backend->call_with_validate(bk_func, bk_results, bk_args);                                 \
        compare_results(new_results, ref_results, bk_results);                                     \
    }                                                                                              \
    ::testing::internal::ParamGenerator<test_case_name::ParamType>                                 \
        gtest_##prefix##test_case_name##_EvalGenerator_()                                          \
    {                                                                                              \
        return generator;                                                                          \
    }                                                                                              \
    ::std::string gtest_##prefix##test_case_name##_EvalGenerateName_(                              \
        const ::testing::TestParamInfo<test_case_name::ParamType>& info)                           \
    {                                                                                              \
        return ::testing::internal::GetParamNameGen<test_case_name::ParamType>()(info);            \
    }                                                                                              \
    static int gtest_##prefix##test_case_name##_dummy_ GTEST_ATTRIBUTE_UNUSED_ =                   \
        ::testing::UnitTest::GetInstance()                                                         \
            ->parameterized_test_registry()                                                        \
            .GetTestCasePatternHolder<test_case_name>(                                             \
                #test_case_name, ::testing::internal::CodeLocation(__FILE__, __LINE__))            \
            ->AddTestCaseInstantiation(#prefix[0] != '\0' ? #prefix : "",                          \
                                       &gtest_##prefix##test_case_name##_EvalGenerator_,           \
                                       &gtest_##prefix##test_case_name##_EvalGenerateName_,        \
                                       __FILE__,                                                   \
                                       __LINE__)
