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

#include <algorithm>
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include "gtest/gtest.h"

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/state/rng_state.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

static const vector<element::Type> s_known_element_types = {element::from<float>(),
                                                            element::from<double>(),
                                                            element::from<int8_t>(),
                                                            element::from<int16_t>(),
                                                            element::from<int32_t>(),
                                                            element::from<int64_t>(),
                                                            element::from<uint8_t>(),
                                                            element::from<uint16_t>(),
                                                            element::from<uint32_t>(),
                                                            element::from<uint64_t>()};

class UnhandledOp : public ngraph::op::Op
{
public:
    UnhandledOp(const std::shared_ptr<Node>& arg)
        : Op("Unsupported_op", check_single_output_args({arg}))
    {
        constructor_validate_and_infer_types();
    }
    shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override
    {
        return make_shared<UnhandledOp>(new_args[0]);
    }

protected:
    void validate_and_infer_types() override
    {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }
};

NGRAPH_TEST(${BACKEND_NAME}, unhandled_op)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor<float>(shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor<float>(shape);
    ASSERT_THROW(auto handle = backend->compile(f);
                 backend->call_with_validate(handle, {result}, {a}), unsupported_op);
}

NGRAPH_TEST(${BACKEND_NAME}, function_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, ParameterVector{A, B}, "funky func name");

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor<float>(shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor<float>(shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor<float>(shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, node_name)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = A + B;
    C->set_name("a node name");
    auto f = make_shared<Function>(C, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, aliased_output)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = A + B;
    auto D = A * B;
    auto E = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
    auto f = make_shared<Function>(NodeVector{C, C, D, D, C, E, E}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out1 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out2 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out3 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out4 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out5 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out6 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> out7 = backend->create_tensor(element::f32, shape);

    copy_data(a, vector<float>{0, 1, 2, 3});
    copy_data(b, vector<float>{1, 2, 3, 4});
    vector<float> expectedC{1, 3, 5, 7};
    vector<float> expectedD{0, 2, 6, 12};
    vector<float> expectedE{1, 2, 3, 4};

    backend->call_with_validate(
        backend->compile(f), {out1, out2, out3, out4, out5, out6, out7}, {a, b});
    EXPECT_EQ(expectedC, read_vector<float>(out1));
    EXPECT_EQ(expectedC, read_vector<float>(out2));
    EXPECT_EQ(expectedD, read_vector<float>(out3));
    EXPECT_EQ(expectedD, read_vector<float>(out4));
    EXPECT_EQ(expectedC, read_vector<float>(out5));
    EXPECT_EQ(expectedE, read_vector<float>(out6));
    EXPECT_EQ(expectedE, read_vector<float>(out7));
}

NGRAPH_TEST(${BACKEND_NAME}, parameter_as_output)
{
    Shape shape{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    vector<float> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<float> zero(shape_size(shape), 0);
    copy_data(a, expected);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, abc)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(handle, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(handle, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, abc_int64)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto C = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{5, 6, 7, 8});
    auto c = backend->create_tensor(element::i64, shape);
    copy_data(c, vector<int64_t>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::i64, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<int64_t>{54, 80, 110, 144}), read_vector<int64_t>(result));

    backend->call_with_validate(handle, {result}, {b, a, c});
    EXPECT_EQ((vector<int64_t>{54, 80, 110, 144}), read_vector<int64_t>(result));

    backend->call_with_validate(handle, {result}, {a, c, b});
    EXPECT_EQ((vector<int64_t>{50, 72, 98, 128}), read_vector<int64_t>(result));
}

// Multiple retrive values
NGRAPH_TEST(${BACKEND_NAME}, multiple_result)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto A_add_B = make_shared<op::Add>(A, B);
    auto A_add_B_mul_C = make_shared<op::Multiply>(A_add_B, C);

    auto f = make_shared<Function>(NodeVector{A_add_B, A_add_B_mul_C}, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{9, 10, 11, 12});

    auto r0 = backend->create_tensor(element::f32, shape);
    auto r1 = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {r0, r1}, {a, b, c});

    EXPECT_EQ((vector<float>{6, 8, 10, 12}), read_vector<float>(r0));
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(r1));
}

template <typename T>
class BatchNormInferenceTester
{
public:
    BatchNormInferenceTester(const std::unique_ptr<ngraph::runtime::Backend>& backend,
                             const Shape& input_shape,
                             element::Type etype,
                             double epsilon)
        : m_backend(backend)
    {
        Shape channel_shape{input_shape.at(1)};

        auto Input = make_shared<op::Parameter>(etype, input_shape);
        auto Gamma = make_shared<op::Parameter>(etype, channel_shape);
        auto Beta = make_shared<op::Parameter>(etype, channel_shape);
        auto Mean = make_shared<op::Parameter>(etype, channel_shape);
        auto Variance = make_shared<op::Parameter>(etype, channel_shape);
        auto BN = make_shared<op::BatchNormInference>(Input, Gamma, Beta, Mean, Variance, epsilon);
        m_function = make_shared<Function>(BN, ParameterVector{Input, Gamma, Beta, Mean, Variance});

        m_input = backend->create_tensor(etype, input_shape);
        m_gamma = backend->create_tensor(etype, channel_shape);
        m_beta = backend->create_tensor(etype, channel_shape);
        m_mean = backend->create_tensor(etype, channel_shape);
        m_variance = backend->create_tensor(etype, channel_shape);
        m_normed_input = backend->create_tensor(etype, input_shape);
    }

    bool call(const std::vector<T>& input,
              const std::vector<T>& gamma,
              const std::vector<T>& beta,
              const std::vector<T>& mean,
              const std::vector<T>& variance,
              const std::vector<T>& normed_input)
    {
        copy_data(m_input, input);
        copy_data(m_gamma, gamma);
        copy_data(m_beta, beta);
        copy_data(m_mean, mean);
        copy_data(m_variance, variance);
        auto handle = m_backend->compile(m_function);
        m_backend->call_with_validate(
            handle, {m_normed_input}, {m_input, m_gamma, m_beta, m_mean, m_variance});
        auto res_normed_input = read_vector<T>(m_normed_input);
        return test::all_close(normed_input, res_normed_input);
    }

protected:
    const std::unique_ptr<ngraph::runtime::Backend>& m_backend;
    std::shared_ptr<Function> m_function;
    std::shared_ptr<ngraph::runtime::Tensor> m_input;
    std::shared_ptr<ngraph::runtime::Tensor> m_gamma;
    std::shared_ptr<ngraph::runtime::Tensor> m_beta;
    std::shared_ptr<ngraph::runtime::Tensor> m_mean;
    std::shared_ptr<ngraph::runtime::Tensor> m_variance;
    std::shared_ptr<ngraph::runtime::Tensor> m_normed_input;
};

template <typename T>
class BatchNormInferenceTesterZeroEpsilon : public BatchNormInferenceTester<T>
{
public:
    // These are for documentation purposes only below
    using Input = test::NDArray<T, 2>;
    using Gamma = test::NDArray<T, 1>;
    using Beta = test::NDArray<T, 1>;
    using Mean = test::NDArray<T, 1>;
    using Variance = test::NDArray<T, 1>;
    using NormedInput = test::NDArray<T, 2>;

    BatchNormInferenceTesterZeroEpsilon(const std::unique_ptr<ngraph::runtime::Backend>& backend,
                                        element::Type etype)
        : BatchNormInferenceTester<T>(backend, Shape{2, 3}, etype, 0.0)
    {
    }

    bool test(const Input& input,
              const Gamma& gamma,
              const Beta& beta,
              const Mean& mean,
              const Variance& variance,
              const NormedInput& normed_input)
    {
        return BatchNormInferenceTester<T>::call(input.get_vector(),
                                                 gamma.get_vector(),
                                                 beta.get_vector(),
                                                 mean.get_vector(),
                                                 variance.get_vector(),
                                                 normed_input.get_vector());
    }

    bool test_gamma()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{2.0, 3.0, 4.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{1.0, 1.0, 1.0},
                    NormedInput{{2.0, 6.0, 12.0}, {-2.0, -6.0, -12.0}});
    }

    bool test_beta()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{2.0, -2.0, 3.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{1.0, 1.0, 1.0},
                    NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
    }

    bool test_mean()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{-2.0, 2.0, -3.0},
                    Variance{1.0, 1.0, 1.0},
                    NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
    }

    bool test_variance()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{0.25, .0625, 4.0},
                    NormedInput{{2.0, 8.0, 1.5}, {-2.0, -8.0, -1.5}});
    }
};

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_0eps_f64)
{
    using T = double;
    auto& et = element::f64;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormInferenceTesterZeroEpsilon<T> bnt(backend, et);
    EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
    EXPECT_TRUE(bnt.test_beta()) << "Beta test";
    EXPECT_TRUE(bnt.test_mean()) << "Mean test";
    EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_0eps_f32)
{
    using T = float;
    auto& et = element::f32;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormInferenceTesterZeroEpsilon<T> bnt(backend, et);
    EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
    EXPECT_TRUE(bnt.test_beta()) << "Beta test";
    EXPECT_TRUE(bnt.test_mean()) << "Mean test";
    EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

template <typename T>
class BatchNormInferenceTesterNonZeroEpsilon : public BatchNormInferenceTester<T>
{
public:
    // These are for documentation purposes only below
    using Input = test::NDArray<T, 2>;
    using Gamma = test::NDArray<T, 1>;
    using Beta = test::NDArray<T, 1>;
    using Mean = test::NDArray<T, 1>;
    using Variance = test::NDArray<T, 1>;
    using NormedInput = test::NDArray<T, 2>;

    BatchNormInferenceTesterNonZeroEpsilon(const std::unique_ptr<ngraph::runtime::Backend>& backend,
                                           element::Type etype)
        : BatchNormInferenceTester<T>(backend, Shape{2, 3}, etype, 0.25)
    {
    }

    bool test(const Input& input,
              const Gamma& gamma,
              const Beta& beta,
              const Mean& mean,
              const Variance& variance,
              const NormedInput& normed_input)
    {
        return BatchNormInferenceTester<T>::call(input.get_vector(),
                                                 gamma.get_vector(),
                                                 beta.get_vector(),
                                                 mean.get_vector(),
                                                 variance.get_vector(),
                                                 normed_input.get_vector());
    }

    bool test_gamma()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{2.0, 3.0, 4.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{0.75, 0.75, 0.75},
                    NormedInput{{2.0, 6.0, 12.0}, {-2.0, -6.0, -12.0}});
    }

    bool test_beta()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{2.0, -2.0, 3.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{0.75, 0.75, 0.75},
                    NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
    }

    bool test_mean()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{-2.0, 2.0, -3.0},
                    Variance{0.75, 0.75, 0.75},
                    NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
    }

    bool test_variance()
    {
        return test(Input{{3.0, 5.0, 1.0}, {-3.0, -5.0, -1.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{2.0, 6.0, 0.0},
                    NormedInput{{2.0, 2.0, 2.0}, {-2.0, -2.0, -2.0}});
    }
};

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_f64)
{
    using T = double;
    auto& et = element::f64;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormInferenceTesterNonZeroEpsilon<T> bnt(backend, et);
    EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
    EXPECT_TRUE(bnt.test_beta()) << "Beta test";
    EXPECT_TRUE(bnt.test_mean()) << "Mean test";
    EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_f32)
{
    using T = float;
    auto& et = element::f32;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormInferenceTesterNonZeroEpsilon<T> bnt(backend, et);
    EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
    EXPECT_TRUE(bnt.test_beta()) << "Beta test";
    EXPECT_TRUE(bnt.test_mean()) << "Mean test";
    EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

template <typename T>
class BatchNormTrainingTester
{
public:
    BatchNormTrainingTester(const std::unique_ptr<ngraph::runtime::Backend>& backend,
                            const Shape& input_shape,
                            element::Type etype,
                            double epsilon)
        : m_backend(backend)
    {
        Shape channel_shape{input_shape.at(1)};

        auto Input = make_shared<op::Parameter>(etype, input_shape);
        auto Gamma = make_shared<op::Parameter>(etype, channel_shape);
        auto Beta = make_shared<op::Parameter>(etype, channel_shape);
        auto BN = make_shared<op::BatchNormTraining>(Input, Gamma, Beta, epsilon);
        auto NormedInput = make_shared<op::Result>(make_shared<op::GetOutputElement>(BN, 0));
        auto Mean = make_shared<op::Result>(make_shared<op::GetOutputElement>(BN, 1));
        auto Variance = make_shared<op::Result>(make_shared<op::GetOutputElement>(BN, 2));
        m_function = make_shared<Function>(ResultVector{NormedInput, Mean, Variance},
                                           ParameterVector{Input, Gamma, Beta});

        m_input = backend->create_tensor(etype, input_shape);
        m_gamma = backend->create_tensor(etype, channel_shape);
        m_beta = backend->create_tensor(etype, channel_shape);
        m_normed_input = backend->create_tensor(etype, input_shape);
        m_mean = backend->create_tensor(etype, channel_shape);
        m_variance = backend->create_tensor(etype, channel_shape);
    }

    std::tuple<bool, bool, bool> call(const std::vector<T>& input,
                                      const std::vector<T>& gamma,
                                      const std::vector<T>& beta,
                                      const std::vector<T>& normed_input,
                                      const std::vector<T>& mean,
                                      const std::vector<T>& variance)
    {
        copy_data(m_input, input);
        copy_data(m_gamma, gamma);
        copy_data(m_beta, beta);
        auto handle = m_backend->compile(m_function);
        m_backend->call_with_validate(
            handle, {m_normed_input, m_mean, m_variance}, {m_input, m_gamma, m_beta});
        auto res_normed_input = read_vector<T>(m_normed_input);
        bool normed_input_test = test::all_close(normed_input, res_normed_input);

        auto res_mean = read_vector<T>(m_mean);
        bool mean_test = test::all_close(mean, res_mean);

        auto res_variance = read_vector<T>(m_variance);
        bool variance_test = test::all_close(variance, res_variance);

        return std::tuple<bool, bool, bool>(normed_input_test, mean_test, variance_test);
    }

protected:
    const std::unique_ptr<ngraph::runtime::Backend>& m_backend;
    std::shared_ptr<Function> m_function;
    std::shared_ptr<ngraph::runtime::Tensor> m_input;
    std::shared_ptr<ngraph::runtime::Tensor> m_gamma;
    std::shared_ptr<ngraph::runtime::Tensor> m_beta;
    std::shared_ptr<ngraph::runtime::Tensor> m_normed_input;
    std::shared_ptr<ngraph::runtime::Tensor> m_mean;
    std::shared_ptr<ngraph::runtime::Tensor> m_variance;
};

template <typename T>
class BatchNormTrainingTesterZeroEpsilon : public BatchNormTrainingTester<T>
{
public:
    // These are for documentation purposes only below
    using Input = test::NDArray<T, 2>;
    using Gamma = test::NDArray<T, 1>;
    using Beta = test::NDArray<T, 1>;
    using NormedInput = test::NDArray<T, 2>;
    using Mean = test::NDArray<T, 1>;
    using Variance = test::NDArray<T, 1>;

    BatchNormTrainingTesterZeroEpsilon(const std::unique_ptr<ngraph::runtime::Backend>& backend,
                                       element::Type etype)
        : BatchNormTrainingTester<T>(backend, Shape{10, 3}, etype, 0.0)
    {
    }

    std::tuple<bool, bool, bool> test(const Input& input,
                                      const Gamma& gamma,
                                      const Beta& beta,
                                      const NormedInput& normed_input,
                                      const Mean& mean,
                                      const Variance& variance)

    {
        return BatchNormTrainingTester<T>::call(input.get_vector(),
                                                gamma.get_vector(),
                                                beta.get_vector(),
                                                normed_input.get_vector(),
                                                mean.get_vector(),
                                                variance.get_vector());
    }

    std::tuple<bool, bool, bool> test_mean_variance()
    {
        return test(Input{{0.0, 1.0, 0.0},
                          {1.0, 2.0, 0.25},
                          {1.0, 2.0, 0.25},
                          {3.0, 4.0, 0.75},
                          {3.0, 4.0, 0.75},
                          {0.0, 1.0, 0.0},
                          {-1.0, 0.0, -0.25},
                          {-1.0, 0.0, -0.25},
                          {-3.0, -2.0, -0.75},
                          {-3.0, -2.0, -0.75}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    NormedInput{{0.0, 0.0, 0.0},
                                {0.5, 0.5, 0.5},
                                {0.5, 0.5, 0.5},
                                {1.5, 1.5, 1.5},
                                {1.5, 1.5, 1.5},
                                {0.0, 0.0, 0.0},
                                {-0.5, -0.5, -0.5},
                                {-0.5, -0.5, -0.5},
                                {-1.5, -1.5, -1.5},
                                {-1.5, -1.5, -1.5}},
                    Mean{0.0, 1.0, 0.0},
                    Variance{4.0, 4.0, 0.25});
    }

    std::tuple<bool, bool, bool> test_gamma_beta()
    {
        return test(Input{{0.0, 1.0, 0.0},
                          {1.0, 2.0, 0.25},
                          {1.0, 2.0, 0.25},
                          {3.0, 4.0, 0.75},
                          {3.0, 4.0, 0.75},
                          {0.0, 1.0, 0.0},
                          {-1.0, 0.0, -0.25},
                          {-1.0, 0.0, -0.25},
                          {-3.0, -2.0, -0.75},
                          {-3.0, -2.0, -0.75}},
                    Gamma{2.0, 1.0, 2.0},
                    Beta{0.0, 1.0, 1.0},
                    NormedInput{{0.0, 1.0, 1.0},
                                {1.0, 1.5, 2.0},
                                {1.0, 1.5, 2.0},
                                {3.0, 2.5, 4.0},
                                {3.0, 2.5, 4.0},
                                {0.0, 1.0, 1.0},
                                {-1.0, 0.5, 0.0},
                                {-1.0, 0.5, 0.0},
                                {-3.0, -0.5, -2.0},
                                {-3.0, -0.5, -2.0}},
                    Mean{0.0, 1.0, 0.0},
                    Variance{4.0, 4.0, 0.25});
    }
};

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_training_0eps_f64)
{
    using T = double;
    auto& et = element::f64;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormTrainingTesterZeroEpsilon<T> bnt(backend, et);
    std::tuple<bool, bool, bool> result;
    result = bnt.test_mean_variance();
    EXPECT_TRUE(std::get<0>(result)) << "Mean variance test normed input";
    EXPECT_TRUE(std::get<1>(result)) << "Mean variance test mean";
    EXPECT_TRUE(std::get<2>(result)) << "Mean variance test variance";

    result = bnt.test_gamma_beta();
    EXPECT_TRUE(std::get<0>(result)) << "Gamma beta test normed input";
    EXPECT_TRUE(std::get<1>(result)) << "Gamma beta test mean";
    EXPECT_TRUE(std::get<2>(result)) << "Gamma test variance";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_training_0eps_f32)
{
    using T = float;
    auto& et = element::f32;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormTrainingTesterZeroEpsilon<T> bnt(backend, et);
    std::tuple<bool, bool, bool> result;
    result = bnt.test_mean_variance();
    EXPECT_TRUE(std::get<0>(result)) << "Mean variance test normed input";
    EXPECT_TRUE(std::get<1>(result)) << "Mean variance test mean";
    EXPECT_TRUE(std::get<2>(result)) << "Mean variance test variance";

    result = bnt.test_gamma_beta();
    EXPECT_TRUE(std::get<0>(result)) << "Gamma beta test normed input";
    EXPECT_TRUE(std::get<1>(result)) << "Gamma beta test mean";
    EXPECT_TRUE(std::get<2>(result)) << "Gamma beta test variance";
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_colwise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 8};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 1),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_rowwise)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_matrix_int64)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    Shape shape_c{3, 2};
    auto C = make_shared<op::Parameter>(element::i64, shape_c);
    Shape shape_r{8, 2};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::i64, shape_b);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::i64, shape_c);
    copy_data(c, vector<int64_t>{2, 3, 5, 7, 11, 13});
    auto result = backend->create_tensor(element::i64, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<int64_t>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              read_vector<int64_t>(result));
}

// Params to drive concat_vector_large testing variations
class concat_vector_params : public ::testing::TestWithParam<int>
{
protected:
    concat_vector_params() { num_inputs = GetParam(); }
    uint32_t num_inputs;
};

NGRAPH_TEST_P(${BACKEND_NAME}, concat_vector_params, concat_vector_large)
{
    Shape shape_a{1};
    NodeVector inputs;
    ParameterVector inputs_param;
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        inputs_param.push_back(A);
        inputs.push_back(A);
    }
    Shape shape_r{num_inputs};
    auto f = make_shared<Function>(make_shared<op::Concat>(inputs, 0), inputs_param);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    std::vector<std::shared_ptr<runtime::Tensor>> inputs_value;
    std::vector<float> ref_result;
    for (uint32_t i = 0; i < num_inputs; i++)
    {
        auto a = backend->create_tensor(element::f32, shape_a);
        copy_data(a, vector<float>{static_cast<float>(i)});
        ref_result.push_back(static_cast<float>(i));
        inputs_value.push_back(a);
    }
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, inputs_value);
    EXPECT_EQ(ref_result, read_vector<float>(result));
}

// concat_vector_large case generation
// Add thhosw tests to cover paramter space overflow:
// cuda kernel parameter space have limit, if there is large number of parameters,
// there will be overflow for parameter space.
NGRAPH_INSTANTIATE_TEST_CASE_P(${BACKEND_NAME},
                               input_sizes,
                               concat_vector_params,
                               testing::Values(100, 128, 999));

NGRAPH_TEST(${BACKEND_NAME}, concat_vector)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{6};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{12};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{18, 19});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_4d_tensor)
{
    Shape shape{1, 1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{3, 1, 1, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_2d_tensor)
{
    Shape shape{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{3, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{A, B, C}, 0),
                                   ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_in_place_2d_tensor)
{
    Shape shape{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto add1 = make_shared<op::Add>(A, B);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto add2 = make_shared<op::Add>(C, D);
    auto subtract = make_shared<op::Subtract>(C, A);
    Shape shape_r{3, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{add1, add2, subtract}, 0),
                                   ParameterVector{A, B, C, D});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto d = backend->create_tensor(element::f32, shape);
    copy_data(d, vector<float>{4});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c, d});
    EXPECT_EQ((vector<float>{3, 7, 2}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_in_place_propagate_2d_tensor)
{
    Shape shape{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto add1 = make_shared<op::Add>(A, B);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto add2 = make_shared<op::Add>(C, D);
    auto concat1 = make_shared<op::Concat>(NodeVector{add1, add2}, 0);
    auto subtract = make_shared<op::Subtract>(C, A);
    Shape shape_r{3, 1};
    auto f = make_shared<Function>(make_shared<op::Concat>(NodeVector{concat1, subtract}, 0),
                                   ParameterVector{A, B, C, D});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto d = backend->create_tensor(element::f32, shape);
    copy_data(d, vector<float>{4});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c, d});
    EXPECT_EQ((vector<float>{3, 7, 2}), read_vector<float>(result));
}

// from numpy import *
// a=linspace(1,2*3*4*3*2,2*3*4*3*2)
// b=linspace(1000+1,1000+2*3*3*3*2,2*3*3*3*2)
// c=linspace(2000+1,2000+2*3*2*3*2,2*3*2*3*2)
// a.shape=(2,3,4,3,2)
// b.shape=(2,3,3,3,2)
// c.shape=(2,3,2,3,2)
// z=concatenate((a,b,c),axis=2)
// z.shape=(2*3*(4+3+2)*3*2)
// set_printoptions(suppress=True)
// print(z)
//
// [    1.     2.     3.     4.     5.     6.     7.     8.     9.    10.
//     11.    12.    13.    14.    15.    16.    17.    18.    19.    20.
//     21.    22.    23.    24.  1001.  1002.  1003.  1004.  1005.  1006.
//   1007.  1008.  1009.  1010.  1011.  1012.  1013.  1014.  1015.  1016.
//   1017.  1018.  2001.  2002.  2003.  2004.  2005.  2006.  2007.  2008.
//   2009.  2010.  2011.  2012.    25.    26.    27.    28.    29.    30.
//     31.    32.    33.    34.    35.    36.    37.    38.    39.    40.
//     41.    42.    43.    44.    45.    46.    47.    48.  1019.  1020.
//   1021.  1022.  1023.  1024.  1025.  1026.  1027.  1028.  1029.  1030.
//   1031.  1032.  1033.  1034.  1035.  1036.  2013.  2014.  2015.  2016.
//   2017.  2018.  2019.  2020.  2021.  2022.  2023.  2024.    49.    50.
//     51.    52.    53.    54.    55.    56.    57.    58.    59.    60.
//     61.    62.    63.    64.    65.    66.    67.    68.    69.    70.
//     71.    72.  1037.  1038.  1039.  1040.  1041.  1042.  1043.  1044.
//   1045.  1046.  1047.  1048.  1049.  1050.  1051.  1052.  1053.  1054.
//   2025.  2026.  2027.  2028.  2029.  2030.  2031.  2032.  2033.  2034.
//   2035.  2036.    73.    74.    75.    76.    77.    78.    79.    80.
//     81.    82.    83.    84.    85.    86.    87.    88.    89.    90.
//     91.    92.    93.    94.    95.    96.  1055.  1056.  1057.  1058.
//   1059.  1060.  1061.  1062.  1063.  1064.  1065.  1066.  1067.  1068.
//   1069.  1070.  1071.  1072.  2037.  2038.  2039.  2040.  2041.  2042.
//   2043.  2044.  2045.  2046.  2047.  2048.    97.    98.    99.   100.
//    101.   102.   103.   104.   105.   106.   107.   108.   109.   110.
//    111.   112.   113.   114.   115.   116.   117.   118.   119.   120.
//   1073.  1074.  1075.  1076.  1077.  1078.  1079.  1080.  1081.  1082.
//   1083.  1084.  1085.  1086.  1087.  1088.  1089.  1090.  2049.  2050.
//   2051.  2052.  2053.  2054.  2055.  2056.  2057.  2058.  2059.  2060.
//    121.   122.   123.   124.   125.   126.   127.   128.   129.   130.
//    131.   132.   133.   134.   135.   136.   137.   138.   139.   140.
//    141.   142.   143.   144.  1091.  1092.  1093.  1094.  1095.  1096.
//   1097.  1098.  1099.  1100.  1101.  1102.  1103.  1104.  1105.  1106.
//   1107.  1108.  2061.  2062.  2063.  2064.  2065.  2066.  2067.  2068.
//   2069.  2070.  2071.  2072.]
NGRAPH_TEST(${BACKEND_NAME}, concat_5d)
{
    vector<float> a_data(2 * 3 * 4 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 4 * 3 * 2; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 3 * 2; i++)
    {
        b_data[i] = 1000 + float(i + 1);
    }

    vector<float> c_data(2 * 3 * 2 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 2 * 3 * 2; i++)
    {
        c_data[i] = 2000 + float(i + 1);
    }

    Shape shape_a{2, 3, 4, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3, 3, 3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3, 2, 3, 2};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 3, 9, 3, 2};

    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 2);
    auto f = make_shared<Function>(r, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, c_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ(
        (vector<float>{
            1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,    10.,   11.,   12.,
            13.,   14.,   15.,   16.,   17.,   18.,   19.,   20.,   21.,   22.,   23.,   24.,
            1001., 1002., 1003., 1004., 1005., 1006., 1007., 1008., 1009., 1010., 1011., 1012.,
            1013., 1014., 1015., 1016., 1017., 1018., 2001., 2002., 2003., 2004., 2005., 2006.,
            2007., 2008., 2009., 2010., 2011., 2012., 25.,   26.,   27.,   28.,   29.,   30.,
            31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,   40.,   41.,   42.,
            43.,   44.,   45.,   46.,   47.,   48.,   1019., 1020., 1021., 1022., 1023., 1024.,
            1025., 1026., 1027., 1028., 1029., 1030., 1031., 1032., 1033., 1034., 1035., 1036.,
            2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021., 2022., 2023., 2024.,
            49.,   50.,   51.,   52.,   53.,   54.,   55.,   56.,   57.,   58.,   59.,   60.,
            61.,   62.,   63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,   72.,
            1037., 1038., 1039., 1040., 1041., 1042., 1043., 1044., 1045., 1046., 1047., 1048.,
            1049., 1050., 1051., 1052., 1053., 1054., 2025., 2026., 2027., 2028., 2029., 2030.,
            2031., 2032., 2033., 2034., 2035., 2036., 73.,   74.,   75.,   76.,   77.,   78.,
            79.,   80.,   81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,   90.,
            91.,   92.,   93.,   94.,   95.,   96.,   1055., 1056., 1057., 1058., 1059., 1060.,
            1061., 1062., 1063., 1064., 1065., 1066., 1067., 1068., 1069., 1070., 1071., 1072.,
            2037., 2038., 2039., 2040., 2041., 2042., 2043., 2044., 2045., 2046., 2047., 2048.,
            97.,   98.,   99.,   100.,  101.,  102.,  103.,  104.,  105.,  106.,  107.,  108.,
            109.,  110.,  111.,  112.,  113.,  114.,  115.,  116.,  117.,  118.,  119.,  120.,
            1073., 1074., 1075., 1076., 1077., 1078., 1079., 1080., 1081., 1082., 1083., 1084.,
            1085., 1086., 1087., 1088., 1089., 1090., 2049., 2050., 2051., 2052., 2053., 2054.,
            2055., 2056., 2057., 2058., 2059., 2060., 121.,  122.,  123.,  124.,  125.,  126.,
            127.,  128.,  129.,  130.,  131.,  132.,  133.,  134.,  135.,  136.,  137.,  138.,
            139.,  140.,  141.,  142.,  143.,  144.,  1091., 1092., 1093., 1094., 1095., 1096.,
            1097., 1098., 1099., 1100., 1101., 1102., 1103., 1104., 1105., 1106., 1107., 1108.,
            2061., 2062., 2063., 2064., 2065., 2066., 2067., 2068., 2069., 2070., 2071., 2072.}),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_zero_length_1d_last)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4};

    auto r = make_shared<op::Concat>(NodeVector{A, B}, 0);
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> a_data{1, 2, 3, 4};
    vector<float> b_data(0);

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_zero_length_1d_middle)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{4};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{8};

    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 0);
    auto f = make_shared<Function>(r, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> a_data{1, 2, 3, 4};
    vector<float> b_data(0);
    vector<float> c_data{5, 6, 7, 8};

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, c_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, concat_zero_length_4d_middle)
{
    Shape shape_a{2, 2, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 0, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 2, 1, 1};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 2, 2, 1};

    auto r = make_shared<op::Concat>(NodeVector{A, B, C}, 2);
    auto f = make_shared<Function>(r, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    vector<float> a_data{1, 2, 3, 4};
    vector<float> b_data(0);
    vector<float> c_data{5, 6, 7, 8};

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, c_data);

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{1, 5, 2, 6, 3, 7, 4, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, lrn)
{
    Shape shape{2, 3, 2, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto lrn = make_shared<op::LRN>(A, 1., 2., 1., 3);
    auto f = make_shared<Function>(lrn, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> args{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, args);

    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});

    vector<float> expected{0.f,
                           0.05325444f,
                           0.03402646f,
                           0.01869806f,
                           0.06805293f,
                           0.03287071f,
                           0.00509002f,
                           0.00356153f,
                           0.00174719f,
                           0.0012555f,
                           0.00322708f,
                           0.00235574f};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, select)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Select>(A, B, C), ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{0, 1, 1, 0, 0, 1, 0, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{11, 12, 13, 14, 15, 16, 17, 18});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{11, 2, 3, 14, 15, 6, 17, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(A, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_2constant)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(NodeVector{A, A}, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result0 = backend->create_tensor(element::f32, shape);
    auto result1 = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result0, result1}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result0));
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_with_op)
{
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {-1, 2, 3, -4, 5, -6, -7, 8});
    auto f = make_shared<Function>(make_shared<op::Abs>(A), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_multi_use)
{
    auto A = make_shared<op::Constant>(element::i32, Shape{}, std::vector<std::string>{"388"});
    auto f = make_shared<Function>(A, ParameterVector{});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::shared_ptr<runtime::Tensor> r1 = backend->create_tensor(element::i32, Shape{});
    backend->call_with_validate(
        backend->compile(f), {r1}, std::vector<std::shared_ptr<runtime::Tensor>>{});
    EXPECT_EQ(read_vector<int>(r1), std::vector<int>{388});

    std::shared_ptr<runtime::Tensor> r2 = backend->create_tensor(element::i32, Shape{});
    backend->call_with_validate(
        backend->compile(f), {r2}, std::vector<std::shared_ptr<runtime::Tensor>>{});
    EXPECT_EQ(read_vector<int>(r2), std::vector<int>{388});
}

NGRAPH_TEST(${BACKEND_NAME}, function_call)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C});

    // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);
    auto Z = make_shared<op::Parameter>(element::f32, shape);
    auto g =
        make_shared<Function>(make_shared<op::FunctionCall>(f, NodeVector{X + Y, Y + Z, Z + X}) +
                                  make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z}),
                              ParameterVector{X, Y, Z});

    // Now call g on some test vectors.
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto x = backend->create_tensor(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->create_tensor(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->create_tensor(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(g);
    backend->call_with_validate(handle, {result}, {x, y, z});
    EXPECT_EQ((vector<float>{254, 368, 502, 656}), read_vector<float>(result));

    backend->call_with_validate(handle, {result}, {y, x, z});
    EXPECT_EQ((vector<float>{278, 400, 542, 704}), read_vector<float>(result));

    backend->call_with_validate(handle, {result}, {x, z, y});
    EXPECT_EQ((vector<float>{194, 296, 418, 560}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_int32_float32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::f32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_uint16_float32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::u16, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::f32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u16, shape);
    copy_data(a, vector<uint16_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_int32_bool)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Convert>(A, element::boolean), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<char>{1, 2, 3, 4}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_float32_bool)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Convert>(A, element::boolean), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<char>{1, 2, 3, 4}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_scalar)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Slice>(A, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{312}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 6, 7, 10, 11}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_vector)
{
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Slice>(A, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_overlap)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto C = make_shared<op::Add>(A, B);
    Shape shape_r{2, 4};
    auto D = make_shared<op::Slice>(C, Coordinate{0, 0}, Coordinate{2, 4});
    auto E = make_shared<op::Slice>(C, Coordinate{1, 0}, Coordinate{3, 4});
    auto r = make_shared<op::Add>(D, E);
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, shape_a);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<float>{12, 16, 20, 24, 28, 32, 36, 40}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_in_place)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 4};
    auto D = make_shared<op::Slice>(A, Coordinate{0, 0}, Coordinate{2, 4});
    auto E = make_shared<op::Slice>(A, Coordinate{2, 0}, Coordinate{4, 4});
    auto r = make_shared<op::Add>(D, E);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(backend->compile(f), {result}, {a});
    EXPECT_EQ((vector<float>{10, 12, 14, 16, 18, 20, 22, 24}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_in_place_twice)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 4};
    auto B = make_shared<op::Slice>(A, Coordinate{0, 0}, Coordinate{2, 4});
    auto D = make_shared<op::Slice>(B, Coordinate{1, 0}, Coordinate{2, 4});
    auto E = make_shared<op::Slice>(A, Coordinate{2, 0}, Coordinate{3, 4});
    auto r = make_shared<op::Add>(D, E);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(backend->compile(f), {result}, {a});
    EXPECT_EQ((vector<float>{14, 16, 18, 20}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_in_place_twice_overlap)
{
    Shape shape_a{5, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 4};
    auto B = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{5, 4});
    auto D = make_shared<op::Slice>(B, Coordinate{1, 0}, Coordinate{3, 4});
    auto E = make_shared<op::Slice>(B, Coordinate{2, 0}, Coordinate{4, 4});
    auto r = make_shared<op::Add>(D, E);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(backend->compile(f), {result}, {a});
    EXPECT_EQ((vector<float>{22, 24, 26, 28, 30, 32, 34, 36}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_in_place_with_reshape)
{
    Shape shape_a{4, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 4};
    auto B = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 5});
    auto C = make_shared<op::Reshape>(B, AxisVector{1, 0}, Shape{5, 3});
    auto D = make_shared<op::Slice>(C, Coordinate{1, 0}, Coordinate{5, 3});
    auto E = make_shared<op::Reshape>(D, AxisVector{1, 0}, Shape{3, 4});
    auto r = make_shared<op::Slice>(E, Coordinate{1, 0}, Coordinate{3, 4});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(backend->compile(f), {result}, {a});
    EXPECT_EQ((vector<float>{12, 13, 14, 15, 17, 18, 19, 20}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_strided)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 4}, Strides{2, 3});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{4, 7, 12, 15}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{0, 2, 8, 10, 32, 34, 40, 42}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided_different_strides)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{0, 3, 8, 11, 32, 35, 40, 43}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, scalar_constant_float32)
{
    auto r = op::Constant::create(element::f32, Shape{}, {4.75});
    auto f = make_shared<Function>(r, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ(vector<float>{4.75f}, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, scalar_constant_int64)
{
    auto r = op::Constant::create(element::i64, Shape{}, {2112});
    auto f = make_shared<Function>(r, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ(vector<int64_t>{2112}, read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_float32)
{
    Shape shape{2, 2};
    auto r = op::Constant::create(element::f32, shape, {4.75, 4.5, -5.25, 0.0});
    auto f = make_shared<Function>(r, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<float>{4.75f, 4.5f, -5.25f, 0.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_int64)
{
    Shape shape{2, 2};
    auto r = op::Constant::create(element::i64, shape, {2112, 1848, 1776, 1964});
    auto f = make_shared<Function>(r, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<int64_t>{2112, 1848, 1776, 1964}), read_vector<int64_t>(result));
}

// TODO: Kahan sum only works in limited cases with CPU / Interpreter backend
NGRAPH_TEST(${BACKEND_NAME}, kahan_sum_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    float epsilon = 9.5367431640625e-7f;
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{epsilon, -1.f, 0.f, 1.f});
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{epsilon}, read_vector<float>(result)));
}

// TODO: Kahan sum only works in limited cases with CPU / Interpreter backend
NGRAPH_TEST(${BACKEND_NAME}, kahan_sum_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    float epsilon_a = 1.220703125e-4f;
    float epsilon_b = 3.0517578125e-5f;
    float epsilon_c = 7.62939453125e-6f;
    copy_data(a, vector<float>{1,  1,  1,  1,  1,  1,  epsilon_a, epsilon_b, epsilon_c,
                               1,  1,  1,  1,  1,  1,  -1,        -1,        -1,
                               -1, -1, -1, -1, -1, -1, -1,        -1,        -1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{epsilon_a, epsilon_b, epsilon_c},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, constant_equality_bool)
{
    Shape shape{4};
    // auto A = make_shared<op::Parameter>(element::boolean, shape);
    // auto B = make_shared<op::Parameter>(element::boolean, shape);
    // auto f = make_shared<Function>(make_shared<op::Equal>(A, B), ParameterVector{A, B});

    auto A = op::Constant::create(element::boolean, shape, {true, false, true, false});
    auto B = op::Constant::create(element::boolean, shape, {true, true, true, true});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<char>{true, false, true, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_scalar)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{808});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<float>{808}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_matrix_inplace)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto abs_A = make_shared<op::Abs>(A);

    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4};
    auto r = make_shared<op::ReplaceSlice>(abs_A, B, Coordinate{0, 1}, Coordinate{3, 3});
    auto abs_r = make_shared<op::Abs>(r);
    auto f = make_shared<Function>(abs_r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 106, 107, 110, 111});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 102, 103, 4, 5, 106, 107, 8, 9, 110, 111, 12, 13, 14, 15, 16}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_matrix)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{3, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 106, 107, 110, 111});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 102, 103, 4, 5, 106, 107, 8, 9, 110, 111, 12, 13, 14, 15, 16}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_vector)
{
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{12};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{16};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ(
        (vector<float>{0, 1, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 14, 15}),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_3d)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{921, 922, 925, 926, 937, 938, 941, 942});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<float>{0,  1,  2,  3,  4,  5,   6,   7,  8,  9,   10,  11, 12, 13, 14, 15,

                             16, 17, 18, 19, 20, 921, 922, 23, 24, 925, 926, 27, 28, 29, 30, 31,

                             32, 33, 34, 35, 36, 937, 938, 39, 40, 941, 942, 43, 44, 45, 46, 47,

                             48, 49, 50, 51, 52, 53,  54,  55, 56, 57,  58,  59, 60, 61, 62, 63}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_3d_strided)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{900, 902, 908, 910, 932, 934, 940, 942});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<float>{900, 1,  902, 3,  4,  5,  6,  7,  908, 9,  910, 11, 12, 13, 14, 15,

                             16,  17, 18,  19, 20, 21, 22, 23, 24,  25, 26,  27, 28, 29, 30, 31,

                             932, 33, 934, 35, 36, 37, 38, 39, 940, 41, 942, 43, 44, 45, 46, 47,

                             48,  49, 50,  51, 52, 53, 54, 55, 56,  57, 58,  59, 60, 61, 62, 63}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, replace_slice_3d_strided_different_strides)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{4, 4, 4};
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{900, 903, 908, 911, 932, 935, 940, 943});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<float>{900, 1,  2,  903, 4,  5,  6,  7,  908, 9,  10, 911, 12, 13, 14, 15,

                             16,  17, 18, 19,  20, 21, 22, 23, 24,  25, 26, 27,  28, 29, 30, 31,

                             932, 33, 34, 935, 36, 37, 38, 39, 940, 41, 42, 943, 44, 45, 46, 47,

                             48,  49, 50, 51,  52, 53, 54, 55, 56,  57, 58, 59,  60, 61, 62, 63}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_0d)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_1d_nochange)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{0, 1, 2, 3, 4, 5, 6, 7}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_1d_0)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{7, 6, 5, 4, 3, 2, 1, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_nochange)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_0)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}).get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_1)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}).get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_2d_01)
{
    Shape shape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 2>({{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}).get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_nochange)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                        {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_0)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}},
                                        {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_1)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}},
                                        {{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_2)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}},
                                        {{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_01)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{21, 22, 23}, {18, 19, 20}, {15, 16, 17}, {12, 13, 14}},
                                        {{9, 10, 11}, {6, 7, 8}, {3, 4, 5}, {0, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_02)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{14, 13, 12}, {17, 16, 15}, {20, 19, 18}, {23, 22, 21}},
                                        {{2, 1, 0}, {5, 4, 3}, {8, 7, 6}, {11, 10, 9}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_12)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}},
                                        {{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_3d_012)
{
    Shape shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Reverse>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}},
                                       {{12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {21, 22, 23}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{23, 22, 21}, {20, 19, 18}, {17, 16, 15}, {14, 13, 12}},
                                        {{11, 10, 9}, {8, 7, 6}, {5, 4, 3}, {2, 1, 0}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_float_nan)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_double_nan)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, NAN, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, NAN});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_float_inf)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f32, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f32, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, numeric_double_inf)
{
    Shape shape{5};
    auto A = op::Constant::create(element::f64, shape, {-2.5f, 25.5f, 2.25f, INFINITY, 6.0f});
    auto B = op::Constant::create(element::f64, shape, {10.0f, 5.0f, 2.25f, 10.0f, -INFINITY});
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::boolean, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<char>{false, false, true, false, false}), read_vector<char>(result));
}

//
// From the XLA docs: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
NGRAPH_TEST(${BACKEND_NAME}, select_and_scatter_with_overlap)
{
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{4, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{4, 5};
    Shape window_shape{2, 3};
    auto window_strides = Strides{2, 2};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>(
                  {{7, 2, 5, 3, 8}, {3, 8, 9, 3, 4}, {1, 5, 7, 5, 6}, {0, 6, 2, 10, 2}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 2>({{2, 6}, {3, 1}}).get_vector());
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((test::NDArray<float, 2>(
                   {{0, 0, 0, 0, 0}, {0, 0, 8, 0, 0}, {0, 0, 3, 0, 0}, {0, 0, 0, 1, 0}})
                   .get_vector()),
              read_vector<float>(result));
}

//
// From the XLA docs: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
NGRAPH_TEST(${BACKEND_NAME}, select_and_scatter_without_overlap)
{
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{4, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{4, 6};
    Shape window_shape{2, 3};
    auto window_strides = Strides{2, 3};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 2>(
                  {{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}, {1, 5, 7, 5, 6, 1}, {0, 6, 2, 7, 2, 8}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 2>({{2, 6}, {3, 1}}).get_vector());
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((test::NDArray<float, 2>(
                   {{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}, {0, 0, 3, 0, 0, 0}, {0, 0, 0, 0, 0, 1}})
                   .get_vector()),
              read_vector<float>(result));
}

//
// Adapted from the XLA docs to provide an example in >2D: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
//
NGRAPH_TEST(${BACKEND_NAME}, select_and_scatter_3d_without_overlap)
{
    Shape shape_sel_a{};
    auto SEL_A = make_shared<op::Parameter>(element::f32, shape_sel_a);
    Shape shape_sel_b{};
    auto SEL_B = make_shared<op::Parameter>(element::f32, shape_sel_b);
    auto sel_f = make_shared<Function>(make_shared<op::Greater>(SEL_A, SEL_B),
                                       ParameterVector{SEL_A, SEL_B});

    Shape shape_scatter_a{};
    auto SCATTER_A = make_shared<op::Parameter>(element::f32, shape_scatter_a);
    Shape shape_scatter_b{};
    auto SCATTER_B = make_shared<op::Parameter>(element::f32, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, ParameterVector{SCATTER_A, SCATTER_B});

    Shape shape_a{2, 4, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{1, 2, 2};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    Shape shape_r{2, 4, 6};
    Shape window_shape{2, 2, 3};
    auto window_strides = Strides{2, 2, 3};
    auto f = make_shared<Function>(
        make_shared<op::SelectAndScatter>(A, B, C, sel_f, scatter_f, window_shape, window_strides),
        ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(
        a,
        test::NDArray<float, 3>(
            {{{7, 2, 5, 3, 10, 2}, {3, 8, 9, 3, 4, 2}, {1, 5, 7, 5, 6, 1}, {0, 6, 2, 7, 2, 8}},
             {{2, 5, 8, 3, 4, 2}, {1, 2, 8, 4, 5, 2}, {10, 2, 3, 4, 1, 0}, {4, 1, 2, 4, 5, 7}}})
            .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, test::NDArray<float, 3>({{{2, 6}, {3, 1}}}).get_vector());
    auto c = backend->create_tensor(element::f32, shape_c);
    copy_data(c, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ(
        (test::NDArray<float, 3>(
             {{{0, 0, 0, 0, 6, 0}, {0, 0, 2, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},
              {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {3, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}})
             .get_vector()),
        read_vector<float>(result));
}

template <typename OP>
void make_unary_empty_test(const string& backend_name)
{
    Shape shape{0};

    ParameterVector params;
    NodeVector result_list;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        shared_ptr<op::Parameter> p = make_shared<op::Parameter>(s_known_element_types[i], shape);
        params.push_back(p);
        result_list.push_back(make_shared<OP>(p));
    }

    auto f = make_shared<Function>(result_list, params);
    auto backend = runtime::Backend::create(backend_name);

    vector<shared_ptr<runtime::Tensor>> inputs;
    vector<shared_ptr<runtime::Tensor>> outputs;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        inputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
        outputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
    }

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(inputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(inputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[9]).size(), 0);

    EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(outputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(outputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(outputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(outputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(outputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(outputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(outputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(outputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(outputs[9]).size(), 0);
}

template <typename OP>
void make_binary_empty_test(const string& backend_name, bool is_comparison = false)
{
    Shape shape{0};
    ParameterVector A;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        A.push_back(make_shared<op::Parameter>(s_known_element_types[i], shape));
    }

    NodeVector result_list;
    for (shared_ptr<op::Parameter> p : A)
    {
        result_list.push_back(make_shared<OP>(p, p));
    }

    auto f = make_shared<Function>(result_list, A);
    auto backend = runtime::Backend::create(backend_name);

    vector<shared_ptr<runtime::Tensor>> inputs;
    vector<shared_ptr<runtime::Tensor>> outputs;
    for (size_t i = 0; i < s_known_element_types.size(); i++)
    {
        inputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
        if (is_comparison)
        {
            outputs.push_back(backend->create_tensor(element::from<char>(), shape));
        }
        else
        {
            outputs.push_back(backend->create_tensor(s_known_element_types[i], shape));
        }
    }

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<double>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int8_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<int16_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[4]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[5]).size(), 0);
    EXPECT_EQ(read_vector<uint8_t>(inputs[6]).size(), 0);
    EXPECT_EQ(read_vector<uint16_t>(inputs[7]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[8]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[9]).size(), 0);

    if (is_comparison)
    {
        EXPECT_EQ(read_vector<char>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[4]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[5]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[6]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[7]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[8]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[9]).size(), 0);
    }
    else
    {
        EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<double>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<int8_t>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<int16_t>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<int32_t>(outputs[4]).size(), 0);
        EXPECT_EQ(read_vector<int64_t>(outputs[5]).size(), 0);
        EXPECT_EQ(read_vector<uint8_t>(outputs[6]).size(), 0);
        EXPECT_EQ(read_vector<uint16_t>(outputs[7]).size(), 0);
        EXPECT_EQ(read_vector<uint32_t>(outputs[8]).size(), 0);
        EXPECT_EQ(read_vector<uint64_t>(outputs[9]).size(), 0);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_abs)
{
    make_unary_empty_test<op::Abs>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_ceiling)
{
    make_unary_empty_test<op::Ceiling>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_exp)
{
    make_unary_empty_test<op::Exp>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_floor)
{
    make_unary_empty_test<op::Floor>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_log)
{
    make_unary_empty_test<op::Log>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_negative)
{
    make_unary_empty_test<op::Negative>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_not)
{
    Shape shape{0};
    auto A = make_shared<op::Parameter>(element::from<char>(), shape);
    auto f = make_shared<Function>(make_shared<op::Not>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::from<char>(), shape);
    auto result = backend->create_tensor(element::from<char>(), shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});

    auto in_vec = read_vector<char>(a);
    auto out_vec = read_vector<char>(result);

    EXPECT_EQ(in_vec.size(), 0);
    EXPECT_EQ(out_vec.size(), 0);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sign)
{
    make_unary_empty_test<op::Sign>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sqrt)
{
    make_unary_empty_test<op::Sqrt>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sin)
{
    make_unary_empty_test<op::Sin>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sinh)
{
    make_unary_empty_test<op::Sinh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_cos)
{
    make_unary_empty_test<op::Cos>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_cosh)
{
    make_unary_empty_test<op::Cosh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_tan)
{
    make_unary_empty_test<op::Tan>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_tanh)
{
    make_unary_empty_test<op::Tanh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_asin)
{
    make_unary_empty_test<op::Asin>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_acos)
{
    make_unary_empty_test<op::Acos>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_atan)
{
    make_unary_empty_test<op::Atan>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_add)
{
    make_binary_empty_test<op::Add>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_divide)
{
    make_binary_empty_test<op::Divide>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_eq)
{
    make_binary_empty_test<op::Equal>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_greater)
{
    make_binary_empty_test<op::Greater>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_greatereq)
{
    make_binary_empty_test<op::GreaterEq>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_less)
{
    make_binary_empty_test<op::Less>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_lesseq)
{
    make_binary_empty_test<op::LessEq>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_maximum)
{
    make_binary_empty_test<op::Maximum>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_minimum)
{
    make_binary_empty_test<op::Minimum>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_multiply)
{
    make_binary_empty_test<op::Multiply>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_not_equal)
{
    make_binary_empty_test<op::NotEqual>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_power)
{
    make_binary_empty_test<op::Power>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_subtract)
{
    make_binary_empty_test<op::Subtract>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_outlining)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 2, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto conv1 = make_shared<op::Convolution>(A,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    auto conv2 = make_shared<op::Convolution>(conv1,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    auto f = make_shared<Function>(conv2, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ(vector<float>{expected_result}, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, computation_reuse)
{
    Shape shape_a{1, 16, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{32, 16, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 32, 2, 2};
    auto conv = make_shared<op::Convolution>(A,
                                             B,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});
    Shape pool_shape{1, 1};
    auto pool = make_shared<op::AvgPool>(conv, pool_shape);
    auto bias = make_shared<op::Broadcast>(
        op::Constant::create(element::f32, Shape{}, {2.14}), shape_r, AxisSet{0, 1, 2, 3});
    auto result_op = make_shared<op::Result>(pool + bias);
    auto f = make_shared<Function>(ResultVector{result_op}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> input(64, 1.0f);
    vector<float> weights(512, 0.5f);
    vector<float> rv(128);

    auto a = backend->create_tensor(element::f32, shape_a, input.data());
    auto b = backend->create_tensor(element::f32, shape_b, weights.data());
    auto result = backend->create_tensor(element::f32, shape_r, rv.data());

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});

    vector<float> rv_saved(rv);

    b->set_stale(false);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ(rv_saved, rv);
}

NGRAPH_TEST(${BACKEND_NAME}, pad_interior_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{16};
    Shape padding_below{0};
    Shape padding_above{0};
    Shape padding_interior{2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 1>(
                   {1, 2112, 2112, 2, 2112, 2112, 3, 2112, 2112, 4, 2112, 2112, 5, 2112, 2112, 6})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{15};
    Shape padding_below{4};
    Shape padding_above{5};
    Shape padding_interior{0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 1>(
                   {2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112, 2112, 2112, 2112, 2112})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_interior_exterior_1d)
{
    Shape shape_a{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{25};
    Shape padding_below{4};
    Shape padding_above{5};
    Shape padding_interior{2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 1>({2112, 2112, 2112, 2112, 1,    2112, 2112, 2, 2112,
                                        2112, 3,    2112, 2112, 4,    2112, 2112, 5, 2112,
                                        2112, 6,    2112, 2112, 2112, 2112, 2112})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_interior_exterior_2d)
{
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{7, 6};
    Shape padding_below{1, 0};
    Shape padding_above{2, 1};
    Shape padding_interior{2, 1};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{9});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{9, 9, 9, 9, 9, 9},
                                        {1, 9, 2, 9, 3, 9},
                                        {9, 9, 9, 9, 9, 9},
                                        {9, 9, 9, 9, 9, 9},
                                        {4, 9, 5, 9, 6, 9},
                                        {9, 9, 9, 9, 9, 9},
                                        {9, 9, 9, 9, 9, 9}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x0)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    Shape padding_below{2, 3};
    Shape padding_above{3, 2};
    Shape padding_interior{0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // copy_data(a, test::NDArray<float, 2>({{}}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x3)
{
    Shape shape_a{0, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    Shape padding_below{2, 1};
    Shape padding_above{3, 1};
    Shape padding_interior{0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_3x0)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{5, 5};
    Shape padding_below{1, 3};
    Shape padding_above{1, 2};
    Shape padding_interior{0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112},
                                        {2112, 2112, 2112, 2112, 2112}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_4d_1x2x2x2)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 4, 4};
    Shape padding_below{0, 0, 1, 1};
    Shape padding_above{0, 0, 1, 1};
    Shape padding_interior{0, 0, 0, 0};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                }
            }
        }).get_vector());
    // clang-format on

    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{42});

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    // clang-format off
    EXPECT_EQ((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                },
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result));
    // clang-format on
}

// This is a regression test for one of TF's unit tests, which was failing.
// The problem was inappropriate handling of the shape computation for a
// zero-length axis with interior padding. Rather than subtract 1 from the
// source shape and multiply by the interior padding (which causes underflow),
// we should just count the pre-interior-padding length as zero.
NGRAPH_TEST(${BACKEND_NAME}, pad_interior_exterior_4d_2x0x3x2)
{
    Shape shape_a{2, 0, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape padding_below{1, 0, 0, 0};
    Shape padding_above{0, 2, 0, 0};
    Shape padding_interior{2, 1, 0, 0};
    Shape shape_r{5, 2, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    // copy_data(a, test::NDArray<float, 2>({}).get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{2112});
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected(5 * 2 * 3 * 2, 2112);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ(expected, read_vector<float>(result));
}

// This test covers the case with multiple image and with asymetric pad
// bug has been found on nvGPU side now covered by this test
NGRAPH_TEST(${BACKEND_NAME}, pad_2channel_2image_asym)
{
    Shape shape_a{2, 2, 4, 4};
    auto window_movement_strides = Strides{2, 2};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 2, 2};
    Shape padding_interior{0, 0, 0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 6, 6};
    auto f = make_shared<Function>(
        make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior),
        ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2}, // img 0 chan 0
                                         {0, 3, 2, 0},
                                         {2, 0, 0, 0},
                                         {0, 2, 1, 0}},

                                        {{0, 0, 0, 2}, // img 0 chan 1
                                         {0, 2, 3, 0},
                                         {2, 0, 1, 0},
                                         {2, 0, 0, 0}}},

                                       {{{0, 2, 1, 1}, // img 1 chan 0
                                         {0, 0, 2, 0},
                                         {0, 0, 1, 2},
                                         {0, 0, 0, 0}},

                                        {{2, 1, 0, 0}, // img 1 chan 1
                                         {0, 2, 0, 0},
                                         {1, 1, 2, 0},
                                         {1, 0, 0, 0}}}})
                  .get_vector());

    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{42});

    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 4>({{{{0, 1, 0, 2, 42, 42}, // img 0 chan 0
                                          {0, 3, 2, 0, 42, 42},
                                          {2, 0, 0, 0, 42, 42},
                                          {0, 2, 1, 0, 42, 42},
                                          {42, 42, 42, 42, 42, 42},
                                          {42, 42, 42, 42, 42, 42}},

                                         {{0, 0, 0, 2, 42, 42}, // img 1 chan 0
                                          {0, 2, 3, 0, 42, 42},
                                          {2, 0, 1, 0, 42, 42},
                                          {2, 0, 0, 0, 42, 42},
                                          {42, 42, 42, 42, 42, 42},
                                          {42, 42, 42, 42, 42, 42}}},

                                        {{{0, 2, 1, 1, 42, 42}, // img 1 chan 0
                                          {0, 0, 2, 0, 42, 42},
                                          {0, 0, 1, 2, 42, 42},
                                          {0, 0, 0, 0, 42, 42},
                                          {42, 42, 42, 42, 42, 42},
                                          {42, 42, 42, 42, 42, 42}},

                                         {{2, 1, 0, 0, 42, 42}, // img 1 chan 1
                                          {0, 2, 0, 0, 42, 42},
                                          {1, 1, 2, 0, 42, 42},
                                          {1, 0, 0, 0, 42, 42},
                                          {42, 42, 42, 42, 42, 42},
                                          {42, 42, 42, 42, 42, 42}}}})
                   .get_vector()),
              read_vector<float>(result));
}

// Trivial case with no reduced axes.
NGRAPH_TEST(${BACKEND_NAME}, product_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, product_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{24}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{15, 48}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{2, 12, 30}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1 * 10 * 19,
                             2 * 11 * 20,
                             3 * 12 * 21,
                             4 * 13 * 22,
                             5 * 14 * 23,
                             6 * 15 * 24,
                             7 * 16 * 25,
                             8 * 17 * 26,
                             9 * 18 * 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1 * 2 * 3,
                             4 * 5 * 6,
                             7 * 8 * 9,
                             10 * 11 * 12,
                             13 * 14 * 15,
                             16 * 17 * 18,
                             19 * 20 * 21,
                             22 * 23 * 24,
                             25 * 26 * 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                             2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                             3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close(vector<float>{1.0f * 10.0f * 9.0f * 4.0f * 13.0f * 6.0f * 7.0f *
                                              12.0f * 3.0f * 2.0f * 11.0f * 8.0f * 5.0f * 14.0f *
                                              5.0f * 8.0f * 11.0f * 2.0f * 3.0f * 12.0f * 7.0f *
                                              6.0f * 13.0f * 4.0f * 9.0f * 10.0f * 1.0f},
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, product_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_2d_to_scalar_int32)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(vector<int32_t>{1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9}, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, product_to_scalar_int32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i32, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{24}), read_vector<int32_t>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<int32_t>{1, 2, 3, 4}), read_vector<int32_t>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, product_to_scalar_int8)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto f = make_shared<Function>(make_shared<op::Product>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int8_t>{24}), read_vector<int8_t>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<int8_t>{1, 2, 3, 4}), read_vector<int8_t>(a));
}

// Trivial case with no reduced axes.
NGRAPH_TEST(${BACKEND_NAME}, max_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_int8)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int8_t>{1, 2, 3, 4}), read_vector<int8_t>(result));
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, max_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_trivial_5d_int32)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{4}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_to_scalar_int8)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int8_t>{4}), read_vector<int8_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{5, 6}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{2, 4, 6}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_int32)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{2, 4, 6}), read_vector<int32_t>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<int32_t>{1, 2, 3, 4, 5, 6}), read_vector<int32_t>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_rows_zero_int32)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{});
    auto result = backend->create_tensor(element::i32, shape_rt);
    copy_data(result, vector<int32_t>({3, 3, 3}));

    int32_t minval = std::numeric_limits<int32_t>::has_infinity
                         ? -std::numeric_limits<int32_t>::infinity()
                         : std::numeric_limits<int32_t>::min();

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{minval, minval, minval}), read_vector<int32_t>(result));
    EXPECT_EQ((vector<int32_t>{}), read_vector<int32_t>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{-std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{25.0f, 26.0f, 27.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{14.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_int32)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{14}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_to_scalar_double)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f64, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape_a);
    copy_data(a, vector<double>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f64, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<double>{14}), read_vector<double>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    float mi = -std::numeric_limits<float>::infinity();

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{mi, mi, mi, mi, mi, mi}), read_vector<float>(result));
}

// Trivial case with no reduced axes.
NGRAPH_TEST(${BACKEND_NAME}, min_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, min_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_trivial_5d_int32)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_to_scalar_int8)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int8_t>{1}), read_vector<int8_t>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<int8_t>{1, 2, 3, 4}), read_vector<int8_t>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 3, 5}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_rows_int32)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 5}), read_vector<int32_t>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<int32_t>{1, 2, 3, 4, 5, 6}), read_vector<int32_t>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity()}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 4, 7, 10, 13, 16, 19, 22, 25}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_scalar_int32)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{1}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    float inf = std::numeric_limits<float>::infinity();

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{inf, inf, inf, inf, inf, inf}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_n1c1h2w2)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto sigmoid_node = make_shared<op::Sigmoid>(input);
    auto func = make_shared<Function>(sigmoid_node, ParameterVector{input});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, input->get_shape());
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, input->get_shape());

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f};
    copy_data(a, dataA);

    auto handle = backend->compile(func);
    backend->call_with_validate(handle, {result}, {a});
    vector<float> expected{0.73105858f, 0.98201379f, 0.73105858f, 0.98201379f};
    ASSERT_TRUE(read_vector<float>(result) == expected);
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_n1c1h4)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto sigmoid_node = make_shared<op::Sigmoid>(input);
    auto func = make_shared<Function>(sigmoid_node, ParameterVector{input});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, input->get_shape());
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, input->get_shape());

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f};
    copy_data(a, dataA);

    auto handle = backend->compile(func);
    backend->call_with_validate(handle, {result}, {a});
    vector<float> expected{0.73105858f, 0.98201379f, 0.73105858f, 0.98201379f};
    ASSERT_TRUE(read_vector<float>(result) == expected);
}

NGRAPH_TEST(${BACKEND_NAME}, sigmoid_bprop_n1c1h4)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto sigmoid_node = make_shared<op::SigmoidBackprop>(input, delta);
    auto func = make_shared<Function>(sigmoid_node, ParameterVector{input, delta});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, input->get_shape());
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, delta->get_shape());
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, input->get_shape());

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{1.0f, 1.0f, 1.0f, 1.0f};

    copy_data(a, dataA);
    copy_data(b, dataB);
    auto handle = backend->compile(func);
    backend->call_with_validate(handle, {result}, {a, b});

    vector<float> expected{0.196612f, 0.0176627f, 0.196612f, 0.0176627f};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dfprop)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::Relu>(A);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_4Dfprop)
{
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::Relu>(A);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, fuse_max_with_constant_zero_input_as_relu)
{
    auto shape_a = Shape{2, 5};
    auto A = op::Constant::create(element::f32, shape_a, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto max = make_shared<op::Maximum>(A, B);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(max, ParameterVector{B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto b = backend->create_tensor(element::f32, shape_a);
    copy_data(b, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {b});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dbackprop)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, ParameterVector{A, delta_val});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto delta = backend->create_tensor(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 2, 0, 4, 0, 6, 7, 0, 9, 0};

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, delta});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, relu_4Dbackprop)
{
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, ParameterVector{A, delta_val});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto delta = backend->create_tensor(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, delta});
    EXPECT_EQ(read_vector<float>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_all)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-3, -2, -1, 0, 1, 2});
    auto result = backend->create_tensor(element::f32, shape);

    auto d = expf(-3) + expf(-2) + expf(-1) + expf(0) + expf(1) + expf(2);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<float> expected{
        expf(-3) / d, expf(-2) / d, expf(-1) / d, expf(0) / d, expf(1) / d, expf(2) / d};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));

    // empty AxisSet is the same as "full" AxisSet
    f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{}), ParameterVector{A});
    backend = runtime::Backend::create("${BACKEND_NAME}");

    auto h1 = backend->compile(f);
    backend->call_with_validate(h1, {result}, {a});
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_3d)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60, -1, -2, -3, -4, -5, -6});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-1);
    auto d1 = expf(-20) + expf(-2);
    auto d2 = expf(-30) + expf(-3);
    auto d3 = expf(-40) + expf(-4);
    auto d4 = expf(-50) + expf(-5);
    auto d5 = expf(-60) + expf(-6);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d1,
                           expf(-30) / d2,
                           expf(-40) / d3,
                           expf(-50) / d4,
                           expf(-60) / d5,
                           expf(-1) / d0,
                           expf(-2) / d1,
                           expf(-3) / d2,
                           expf(-4) / d3,
                           expf(-5) / d4,
                           expf(-6) / d5};

    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_3d_double)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f64, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f64, shape);
    copy_data(a, vector<double>{-10, -20, -30, -40, -50, -60, -1, -2, -3, -4, -5, -6});
    auto result = backend->create_tensor(element::f64, shape);

    auto d0 = expf(-10) + expf(-1);
    auto d1 = expf(-20) + expf(-2);
    auto d2 = expf(-30) + expf(-3);
    auto d3 = expf(-40) + expf(-4);
    auto d4 = expf(-50) + expf(-5);
    auto d5 = expf(-60) + expf(-6);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<double> expected{expf(-10) / d0,
                            expf(-20) / d1,
                            expf(-30) / d2,
                            expf(-40) / d3,
                            expf(-50) / d4,
                            expf(-60) / d5,
                            expf(-1) / d0,
                            expf(-2) / d1,
                            expf(-3) / d2,
                            expf(-4) / d3,
                            expf(-5) / d4,
                            expf(-6) / d5};

    EXPECT_TRUE(test::all_close(expected, read_vector<double>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-20) + expf(-30);
    auto d1 = expf(-40) + expf(-50) + expf(-60);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d0,
                           expf(-30) / d0,
                           expf(-40) / d1,
                           expf(-50) / d1,
                           expf(-60) / d1};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_2)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-40);
    auto d1 = expf(-20) + expf(-50);
    auto d2 = expf(-30) + expf(-60);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d1,
                           expf(-30) / d2,
                           expf(-40) / d0,
                           expf(-50) / d1,
                           expf(-60) / d2};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_3d_trivial)
{
    Shape shape{1, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<float> expected{1, 1, 1, 1, 1, 1};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_underflow)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto low = std::numeric_limits<float>::lowest();

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{low, 1, 2, 3, 4, 5});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(low) + expf(3);
    auto d1 = expf(1) + expf(4);
    auto d2 = expf(2) + expf(5);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<float> expected{
        expf(low) / d0, expf(1) / d1, expf(2) / d2, expf(3) / d0, expf(4) / d1, expf(5) / d2};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_overflow)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Softmax>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto high = std::numeric_limits<float>::max();

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{high, 1, 2, 3, 4, 5});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(high - high) + expf(3 - high);
    auto d1 = expf(1) + expf(4);
    auto d2 = expf(2) + expf(5);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<float> expected{expf(high - high) / d0,
                           expf(1) / d1,
                           expf(2) / d2,
                           expf(3 - high) / d0,
                           expf(4) / d1,
                           expf(5) / d2};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, multiple_backends)
{
    Shape shape{2, 2};
    auto A1 = make_shared<op::Parameter>(element::f32, shape);
    auto B1 = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A1 + B1, ParameterVector{A1, B1});

    auto A2 = make_shared<op::Parameter>(element::f32, shape);
    auto B2 = make_shared<op::Parameter>(element::f32, shape);
    auto g = make_shared<Function>(A2 * B2, ParameterVector{A2, B2});

    auto backend1 = runtime::Backend::create("${BACKEND_NAME}");

    auto backend2 = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a1 = backend1->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b1 = backend1->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result1 = backend1->create_tensor(element::f32, shape);

    shared_ptr<runtime::Tensor> a2 = backend2->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b2 = backend2->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result2 = backend2->create_tensor(element::f32, shape);

    copy_data(a1, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b1, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    copy_data(a2, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b2, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend1->call_with_validate(backend1->compile(f), {result1}, {a1, b1});
    EXPECT_EQ(read_vector<float>(result1),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());

    backend2->call_with_validate(backend2->compile(g), {result2}, {a2, b2});
    EXPECT_EQ(read_vector<float>(result2),
              (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, tensorview_custom_mem)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

        return f;
    };

    auto f = make_external();

    vector<float> av{2, 4, 8, 16};
    vector<float> bv{1, 2, 4, 8};
    // use custom mem with tensorview, no need to copy data
    auto a = backend->create_tensor(element::f32, shape, av.data());
    auto b = backend->create_tensor(element::f32, shape, bv.data());

    // use custom mem with result tensorview
    vector<float> rv{0, 0, 0, 0};
    auto result = backend->create_tensor(element::f32, shape, rv.data());

    // result should be in memory without needing explict read
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), rv);
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_count)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f);
                     backend->call_with_validate(handle, {c}, {a}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_type)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f);
                     backend->call_with_validate(handle, {c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_input_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f);
                     backend->call_with_validate(handle, {c}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_count)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);
    auto d = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f);
                     backend->call_with_validate(handle, {c, d}, {a, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_type)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::i32, shape);
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f);
                     backend->call_with_validate(handle, {a}, {b, c}));
}

NGRAPH_TEST(${BACKEND_NAME}, validate_call_output_shape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto a = backend->create_tensor(element::f32, {2, 3});
    auto b = backend->create_tensor(element::f32, shape);
    auto c = backend->create_tensor(element::f32, shape);

    EXPECT_ANY_THROW(auto handle = backend->compile(f);
                     backend->call_with_validate(handle, {a}, {c, b}));
}

NGRAPH_TEST(${BACKEND_NAME}, logical_and)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::And>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 1, 1, 1, 0, 1, 0});
    auto b = backend->create_tensor(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 1, 0, 0, 1, 1, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 0, 1, 0, 0, 0, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, logical_or)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::Or>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 1, 1, 1, 0, 1, 0});
    auto b = backend->create_tensor(element::boolean, shape);
    copy_data(b, vector<char>{0, 0, 1, 0, 0, 1, 1, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<char>{1, 0, 1, 1, 1, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fprop_b1c2h2w2)
{
    auto input_shape = Shape{1, 2, 2, 2};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = make_shared<op::BatchNormTraining>(input, gamma, beta, eps);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   ParameterVector{input, gamma, beta});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});

    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});
    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    vector<float> expected_result{-0.71498716f,
                                  1.48388731f,
                                  -0.00196938f,
                                  -0.76693159f,
                                  -0.91316032f,
                                  0.23943391f,
                                  -0.84090298f,
                                  1.51462936f};
    vector<float> expected_mean{0.602912f, 0.599727f};
    vector<float> expected_variance{0.00472505f, 0.0361782f};

    backend->call_with_validate(
        backend->compile(f), {bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output), 1e-5f, 1e-6f));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean), 1e-5f, 1e-6f));
    EXPECT_TRUE(
        test::all_close(expected_variance, read_vector<float>(result_variance), 1e-5f, 1e-6f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fprop_b2c2h2w1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNormTraining>(input, gamma, beta, eps);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   ParameterVector{input, gamma, beta});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    vector<float> expected_mean{0.583388f, 0.619252f};
    vector<float> expected_variance{0.0119972f, 0.0282681f};
    backend->call_with_validate(
        backend->compile(f), {bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output)));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean)));
    EXPECT_TRUE(
        test::all_close(expected_variance, read_vector<float>(result_variance), 1e-5f, 1e-6f));
}

NGRAPH_TEST(${BACKEND_NAME}, batchnorm_fprop_b2c2d2h1w1)
{
    auto input_shape = Shape{2, 2, 2, 1, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1, 1};
    auto bn = make_shared<op::BatchNormTraining>(eps, gamma, beta, input);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   ParameterVector{input, gamma, beta});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    vector<float> expected_mean{0.583388f, 0.619252f};
    vector<float> expected_variance{0.0119972f, 0.0282681f};
    backend->call_with_validate(
        backend->compile(f), {bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output)));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean)));
    EXPECT_TRUE(
        test::all_close(expected_variance, read_vector<float>(result_variance), 1e-5f, 1e-6f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_bprop_n4c3h2w2)
{
    auto input_shape = Shape{4, 3, 2, 2};
    auto shape_mean = Shape{3};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{3};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{3};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{3};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{3};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{4, 3, 2, 2};
    auto bn = make_shared<op::BatchNormTraining>(input, gamma, beta, eps);
    auto bn_dx = make_shared<op::GetOutputElement>(bn, 0);
    auto bn_dgamma = make_shared<op::GetOutputElement>(bn, 1);
    auto bn_dbeta = make_shared<op::GetOutputElement>(bn, 2);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto _input = backend->create_tensor(element::f32, input_shape);
    vector<float> dataInput{
        10.76331902f, 11.51178265f, 10.31018162f, 12.2993021f,  14.17626667f, 14.63498497f,
        13.63494492f, 13.84248161f, 11.34602547f, 13.22014618f, 10.46686649f, 10.39842987f,
        12.94806862f, 11.71670246f, 14.94438076f, 13.13236618f, 13.40889645f, 12.76128387f,
        11.34430027f, 11.86629677f, 11.11464024f, 10.93221283f, 11.95324039f, 10.96581173f,
        13.05455494f, 14.41404247f, 13.11169434f, 11.26559448f, 10.89965153f, 14.08202171f,
        11.12685776f, 12.58428574f, 12.59247875f, 13.00187492f, 12.66310215f, 10.06655025f,
        12.62048626f, 14.47942352f, 13.84950638f, 10.61425877f, 11.47936344f, 13.06011772f,
        13.63069057f, 12.31748772f, 13.84555244f, 10.95815468f, 12.78933334f, 12.75389099f};
    copy_data(_input, dataInput);
    auto _mean = backend->create_tensor(element::f32, mean_shape);
    copy_data(_mean, vector<float>{12.56472874f, 12.80312157f, 11.81676865f});
    auto _var = backend->create_tensor(element::f32, var_shape);
    copy_data(_var, vector<float>{1.94557643f, 1.32772446f, 1.28163588f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{2.0f, 2.0f, 2.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{1.0f, 1.0f, 1.0f});
    auto result = backend->create_tensor(element::f32, shape_r);

    shared_ptr<runtime::Tensor> _delta = backend->create_tensor(element::f32, shape_r);
    vector<float> deltaData(shape_size(shape_r), 20.0f);
    copy_data(_delta, deltaData);

    auto f = make_shared<Function>(NodeVector{bn_dx, bn_dgamma, bn_dbeta},
                                   ParameterVector{mean, var, input, gamma, beta});

    auto C = std::make_shared<op::Parameter>(element::f32, shape_r);

    auto zero = ngraph::make_zero(bn_dgamma->get_element_type(), bn_dgamma->get_shape());
    ngraph::autodiff::Adjoints adjoints(NodeVector{bn_dx, bn_dgamma, bn_dbeta},
                                        NodeVector{C, zero, zero});

    auto dinput = adjoints.backprop_node(input);
    auto dgamma = adjoints.backprop_node(gamma);
    auto dbeta = adjoints.backprop_node(beta);

    auto df = make_shared<Function>(NodeVector{dinput, dgamma, dbeta},
                                    ParameterVector{mean, var, input, gamma, beta, C});

    // roundtrip serialization
    string js = serialize(df, 4);
    istringstream in(js);
    df = deserialize(in);

    shared_ptr<runtime::Tensor> _dinput = backend->create_tensor(element::f32, shape_r);
    shared_ptr<runtime::Tensor> _dgamma = backend->create_tensor(element::f32, gamma_shape);
    shared_ptr<runtime::Tensor> _dbeta = backend->create_tensor(element::f32, beta_shape);

    auto handle = backend->compile(df);
    backend->call_with_validate(
        handle, {_dinput, _dgamma, _dbeta}, {_mean, _var, _input, _gamma, _beta, _delta});

    vector<float> expected_input{
        8.17051607e-06f,  4.77576657e-06f,  1.02257760e-05f,  1.20387525e-06f,  -1.73868522e-06f,
        3.84632768e-06f,  -1.07932050e-05f, -2.57458956e-06f, -2.22166714e-06f, -8.38779043e-06f,
        -2.48082982e-06f, 5.89238360e-06f,  -2.52895109e-07f, -8.68433445e-06f, -5.82726737e-06f,
        8.84659658e-06f,  3.03944108e-05f,  4.05480879e-05f,  1.84123158e-05f,  2.30061178e-05f,
        1.34087590e-05f,  -9.26072571e-07f, -3.22908454e-05f, -2.07365116e-05f, -4.21330941e-05f,
        2.83083100e-05f,  -3.71039101e-05f, -4.84390640e-06f, -2.93012376e-05f, 5.68858087e-06f,
        1.83181458e-05f,  -1.07494506e-05f, -2.32429103e-06f, 6.92914809e-06f,  -6.66512321e-06f,
        -7.00302840e-06f, -3.46675184e-06f, -4.36748381e-06f, 6.73822226e-07f,  -4.20158993e-06f,
        3.83005061e-06f,  5.85143729e-06f,  4.17875243e-06f,  -8.64167783e-06f, 1.00170803e-05f,
        -4.23939666e-06f, 4.80201680e-06f,  4.62702078e-06f};

    ASSERT_TRUE(ngraph::test::all_close(read_vector<float>(_dinput), expected_input, 1e-3f, 1e-4f));
    vector<float> expected_dgamma{7.06315041e-05f, -2.35289335e-04f, -5.06639481e-05f};
    ASSERT_TRUE(
        ngraph::test::all_close(read_vector<float>(_dgamma), expected_dgamma, 1e-2f, 1e-3f));
    vector<float> expected_dbeta{320.f, 320.f, 320.f};
    ASSERT_TRUE(ngraph::test::all_close(read_vector<float>(_dbeta), expected_dbeta, 1e-4f, 1e-8f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fprop_inference_b2c2h2w1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNormInference>(input, gamma, beta, mean, var, eps);

    auto f = make_shared<Function>(bn, ParameterVector{input, gamma, beta, mean, var});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto _mean = backend->create_tensor(element::f32, mean_shape);
    copy_data(_mean, vector<float>{0.583388f, 0.619252f});
    auto _var = backend->create_tensor(element::f32, var_shape);
    copy_data(_var, vector<float>{0.0119972f, 0.0282681f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    backend->call_with_validate(
        backend->compile(f), {bn_output}, {_input, _gamma, _beta, _mean, _var});

    ASSERT_TRUE(
        ngraph::test::all_close(expected_result, read_vector<float>(bn_output), 1e-3f, 1e-4f));
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_n2c3h4w2)
{
    Shape shape{2, 3, 4, 2};
    Shape seq_len_shape{4};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 2;
    size_t sequence_axis = 1;
    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input{
        0,  0, 3,  0, 6,  0, 9,  0, 1,  0, 4,  0, 7,  0, 10, 0, 2,  0, 5,  0, 8,  0, 11, 0,
        12, 0, 15, 0, 18, 0, 21, 0, 13, 0, 16, 0, 19, 0, 22, 0, 14, 0, 17, 0, 20, 0, 23, 0,
    };

    std::vector<int> seq_lenghts{1, 2, 1, 2};
    copy_data(b, seq_lenghts);

    std::vector<int> expected{
        0,  0, 4,  0, 6,  0, 10, 0, 1,  0, 3,  0, 7,  0, 9,  0, 2,  0, 5,  0, 8,  0, 11, 0,

        12, 0, 16, 0, 18, 0, 22, 0, 13, 0, 15, 0, 19, 0, 21, 0, 14, 0, 17, 0, 20, 0, 23, 0};

    copy_data(a, input);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_n4c3h2w2)
{
    Shape shape{4, 3, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{4};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 0;
    size_t sequence_axis = 1;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> seq_lenghts{1, 2, 3, 3};
    copy_data(b, seq_lenghts);

    std::vector<int> input{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};

    std::vector<int> expected{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17, 18, 19,
                              12, 13, 14, 15, 20, 21, 22, 23, 32, 33, 34, 35, 28, 29, 30, 31,
                              24, 25, 26, 27, 44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39};

    copy_data(a, input);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, reverse_sequence_n4d2c3h2w2)
{
    Shape shape{4, 2, 3, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{4};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 0;
    size_t sequence_axis = 2;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                           48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                           64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};

    std::vector<int> expected{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27,
                              32, 33, 34, 35, 40, 41, 42, 43, 36, 37, 38, 39, 44, 45, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                              64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 72, 73, 74, 75,
                              80, 81, 82, 83, 88, 89, 90, 91, 84, 85, 86, 87, 92, 93, 94, 95};

    copy_data(a, input);

    std::vector<int> seq_lenghts{1, 2, 1, 2};
    copy_data(b, seq_lenghts);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

NGRAPH_TEST(${BACKEND_NAME}, generate_mask)
{
    Shape scalar{};
    Shape result_shape{1, 128};
    const unsigned int seed = 777;
    auto training = op::Constant::create(element::f32, Shape{}, {1});
    auto gen_mask = make_shared<op::GenerateMask>(training, result_shape, element::f32, seed, 0.5);
    auto gen_mask2 = make_shared<op::GenerateMask>(training, result_shape, element::f32, seed, 0.5);
    auto f = make_shared<Function>(NodeVector{gen_mask, gen_mask2}, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto is_not_zero_or_one = [](float num) { return num != 0.f && num != 1.f; };

    auto result_tv1 = backend->create_tensor<float>(result_shape);
    auto result_tv2 = backend->create_tensor<float>(result_shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result_tv1, result_tv2}, {});
    auto result1 = read_vector<float>(result_tv1);
    auto result2 = read_vector<float>(result_tv2);
    ASSERT_EQ(result1, result2);
    ASSERT_FALSE(std::any_of(result1.begin(), result1.end(), is_not_zero_or_one));
    backend->call_with_validate(handle, {result_tv1, result_tv2}, {});
    auto result1_2 = read_vector<float>(result_tv1);
    auto result2_2 = read_vector<float>(result_tv2);
    ASSERT_NE(result1, result1_2);
    ASSERT_FALSE(std::any_of(result1_2.begin(), result1_2.end(), is_not_zero_or_one));
    ASSERT_NE(result2, result2_2);
    ASSERT_FALSE(std::any_of(result2_2.begin(), result2_2.end(), is_not_zero_or_one));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {1});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // divide by scale                2  2  2  2  2  2  2  2  2  2  2   2
    // equals (rounded)               0  0  1  2  2  2  3  4  4  4  5   6
    // plus offset                    1  1  1  1  1  1  1  1  1  1  1   1
    // equals                         1  1  2  3  3  3  4  5  5  5  6   7

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::u8;
    auto output_type = element::f32;

    typedef uint8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {1});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7});
    // minus offset                   1  1  1  1  1  1  1  1  1  1  1  1
    // eqauls                         0  0  1  2  2  2  3  4  4  4  5  6
    // multiplied by scale            2  2  2  2  2  2  2  2  2  2  2  2
    // equals                         0  0  2  4  4  4  6  8  8  8 10 12

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 2, 4, 4, 4, 6, 8, 8, 8, 10, 12}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // divide by scale                2  2  2  2  2  2  2  2  2  2  2   2
    // equals (rounded)               0  0  1  2  2  2  3  4  4  4  5   6
    // plus offset                    0  0  0  0  0  0  0  0  0  0  0   0
    // equals                         0  0  1  2  2  2  3  4  4  4  5   6

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::u8;
    auto output_type = element::f32;

    typedef uint8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {0});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6});
    // minus offset                   0  0  0  0  0  0  0  0  0  0  0  0
    // equals                         0  0  1  2  2  2  3  4  4  4  5  6
    // multiplied by scale            2  2  2  2  2  2  2  2  2  2  2  2
    // equals                         0  0  2  4  4  4  6  8  8  8 10 12

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 2, 4, 4, 4, 6, 8, 8, 8, 10, 12}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_axes)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape{4};
    AxisSet quantization_axes{0};

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2, 3, 4, 5});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {10, 20, 30, 40});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // divided by scale               2  2  2  3  3  3  4  4  4  5  5   5
    // equals (rounded)               0  1  1  1  1  2  2  2  2  2  2   2
    // plus offset                   10 10 10 20 20 20 30 30 30 40 40  40
    // equals                        10 11 11 21 21 22 32 32 32 42 42  42

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{10, 11, 11, 21, 21, 22, 32, 32, 32, 42, 42, 42}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_axes)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape{4};
    AxisSet quantization_axes{0};

    auto input_type = element::u8;
    auto output_type = element::f32;

    typedef uint8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2, 3, 4, 5});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {10, 20, 30, 40});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{10, 11, 11, 21, 21, 22, 32, 32, 32, 42, 42, 42});
    // minus offset                   10  10  10  20  20  20  30  30  30  40  40  40
    // equals                          0   1   1   1   1   2   2   2   2   2   2   2
    // multiplied by scale             2   2   2   3   3   3   4   4   4   5   5   5
    // equals                          0   2   2   3   3   6   8   8   8  10  10  10

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 2, 2, 3, 3, 6, 8, 8, 8, 10, 10, 10}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_int8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {1});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});
    // divide by scale                2   2  2   2  2   2  2   2  2   2  2    2
    // equals (rounded)               0   0  1  -2  2  -2  3  -4  4  -4  5   -6
    // plus offset                    1   1  1   1  1   1  1   1  1   1  1    1
    // equals                         1   1  2  -1  3  -1  4  -3  5  -3  6   -5

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{1, 1, 2, -1, 3, -1, 4, -3, 5, -3, 6, -5}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_int8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::i8;
    auto output_type = element::f32;

    typedef int8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {1});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{1, 1, 2, -1, 3, -1, 4, -3, 5, -3, 6, -5});
    // minus offset                   1  1  1   1  1   1  1   1  1   1  1   1
    // equals                         0  0  1  -2  2  -2  3  -4  4  -4  5  -6
    // multiplied by scale            2  2  2   2  2   2  2   2  2   2  2   2
    // equals                         0  0  2  -4  4  -4  6  -8  8  -8 10 -12

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 2, -4, 4, -4, 6, -8, 8, -8, 10, -12}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_int8_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});
    // divide by scale                2   2  2   2  2   2  2   2  2   2  2    2
    // equals (rounded)               0   0  1  -2  2  -2  3  -4  4  -4  5   -6
    // plus offset                    0   0  0   0  0   0  0   0  0   0  0    0
    // equals                         0   0  1  -2  2  -2  3  -4  4  -4  5   -6

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 1, -2, 2, -2, 3, -4, 4, -4, 5, -6}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_int8_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::i8;
    auto output_type = element::f32;

    typedef int8_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {0});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 0, 1, -2, 2, -2, 3, -4, 4, -4, 5, -6});
    // minus offset                   0  0  0   0  0   0  0   0  0   0  0   0
    // equals                         0  0  1  -2  2  -2  3  -4  4  -4  5  -6
    // multiplied by scale            2  2  2   2  2   2  2   2  2   2  2   2
    // equals                         0  0  2  -4  4  -4  6  -8  8  -8 10 -12

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 2, -4, 4, -4, 6, -8, 8, -8, 10, -12}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_int32)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i32;

    typedef float input_c_type;
    typedef int32_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {1});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});
    // divide by scale                2   2  2   2  2   2  2   2  2   2  2    2
    // equals (rounded)               0   0  1  -2  2  -2  3  -4  4  -4  5   -6
    // plus offset                    1   1  1   1  1   1  1   1  1   1  1    1
    // equals                         1   1  2  -1  3  -1  4  -3  5  -3  6   -5

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{1, 1, 2, -1, 3, -1, 4, -3, 5, -3, 6, -5}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_int32)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::i32;
    auto output_type = element::f32;

    typedef int32_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {1});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{1, 1, 2, -1, 3, -1, 4, -3, 5, -3, 6, -5});
    // minus offset                   1  1  1   1  1   1  1   1  1   1  1   1
    // equals                         0  0  1  -2  2  -2  3  -4  4  -4  5  -6
    // multiplied by scale            2  2  2   2  2   2  2   2  2   2  2   2
    // equals                         0  0  2  -4  4  -4  6  -8  8  -8 10 -12

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 2, -4, 4, -4, 6, -8, 8, -8, 10, -12}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_int32_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i32;

    typedef float input_c_type;
    typedef int32_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});
    // divide by scale                2   2  2   2  2   2  2   2  2   2  2    2
    // equals (rounded)               0   0  1  -2  2  -2  3  -4  4  -4  5   -6
    // plus offset                    0   0  0   0  0   0  0   0  0   0  0    0
    // equals                         0   0  1  -2  2  -2  3  -4  4  -4  5   -6

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 1, -2, 2, -2, 3, -4, 4, -4, 5, -6}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, dequantize_int32_zero_offset)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::i32;
    auto output_type = element::f32;

    typedef int32_t input_c_type;
    typedef float output_c_type;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(input_type, scale_offset_shape, {0});
    auto dequantize = make_shared<op::Dequantize>(X, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 0, 1, -2, 2, -2, 3, -4, 4, -4, 5, -6});
    // minus offset                   0  0  0   0  0   0  0   0  0   0  0   0
    // equals                         0  0  1  -2  2  -2  3  -4  4  -4  5  -6
    // multiplied by scale            2  2  2   2  2   2  2   2  2   2  2   2
    // equals                         0  0  2  -4  4  -4  6  -8  8  -8 10 -12

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, 0, 2, -4, 4, -4, 6, -8, 8, -8, 10, -12}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_clamp_uint8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::u8;

    typedef float input_c_type;
    typedef uint8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto max = std::numeric_limits<uint8_t>::max();

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {1.0 / (max + 1.0)});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, max, max, max, max, max, max, max, max, max, max, max}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_clamp_int8)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto min = std::numeric_limits<int8_t>::min();
    auto max = std::numeric_limits<int8_t>::max();

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {1.0 / (max + 1.0)});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, min, max, min, max, min, max, min, max, min, max, min}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_clamp_int32)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f64;
    auto output_type = element::i32;

    // TODO: fails with input due to 32 bits
    typedef double input_c_type;
    typedef int32_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto min = std::numeric_limits<int32_t>::min();
    auto max = std::numeric_limits<int32_t>::max();

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {1.0 / (max + 1.0)});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{0, min, max, min, max, min, max, min, max, min, max, min}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_TOWARD_ZERO)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_ZERO;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   3  -2   -2   -3   3   3   4   -3   -3   -4

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 3, -2, -2, -3, 3, 3, 4, -3, -3, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_TOWARD_INFINITY)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   3   3  -2   -3   -3   3   4   4   -3   -4   -4

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 3, 3, -2, -3, -3, 3, 4, 4, -3, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_UPWARD)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_UPWARD;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   3   3  -2   -2   -3   3   4   4   -3   -3   -4

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 3, 3, -2, -2, -3, 3, 4, 4, -3, -3, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_DOWNWARD)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_DOWNWARD;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   3  -2   -3   -3   3   3   4   -3   -4   -4

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 3, -2, -3, -3, 3, 3, 4, -3, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_NEAREST_TOWARD_EVEN)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   3  -2   -2   -3   3   4   4   -3   -4   -4

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 3, -2, -2, -3, 3, 4, 4, -3, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_TOWARD_INFINITY)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_TOWARD_INFINITY;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize = make_shared<op::Quantize>(
        X,
        scale,
        offset,
        output_type,
        quantization_axes,
        static_cast<op::Quantize::RoundMode>(static_cast<int>(round_mode)));
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               3   3   3  -3   -3   -3   4   4   4   -4   -4   -4

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{3, 3, 3, -3, -3, -3, 4, 4, 4, -4, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_TOWARD_ZERO)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_TOWARD_ZERO;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize = make_shared<op::Quantize>(
        X,
        scale,
        offset,
        output_type,
        quantization_axes,
        static_cast<op::Quantize::RoundMode>(static_cast<int>(round_mode)));
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   2  -2   -2   -2   3   3   3   -3   -3   -3

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 2, -2, -2, -2, 3, 3, 3, -3, -3, -3}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_UP)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_UP;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               3   3   3  -2   -2   -2   4   4   4   -3   -3   -3

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{3, 3, 3, -2, -2, -2, 4, 4, 4, -3, -3, -3}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, quantize_ROUND_DOWN)
{
    Shape input_shape{4, 3};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto input_type = element::f32;
    auto output_type = element::i8;

    typedef float input_c_type;
    typedef int8_t output_c_type;

    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_DOWN;

    auto X = make_shared<op::Parameter>(input_type, input_shape);
    auto scale = op::Constant::create(input_type, scale_offset_shape, {4});
    auto offset = op::Constant::create(output_type, scale_offset_shape, {0});
    auto quantize =
        make_shared<op::Quantize>(X, scale, offset, output_type, quantization_axes, round_mode);
    auto f = make_shared<Function>(quantize, ParameterVector{X});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto x = backend->create_tensor(input_type, input_shape);
    auto y = backend->create_tensor(output_type, input_shape);

    copy_data(x, vector<input_c_type>{9, 10, 11, -9, -10, -11, 13, 14, 15, -13, -14, -15});
    // divide by scale                4   4   4   4    4    4   4   4   4    4    4    4
    // equals (rounded)               2   2   2  -3   -3   -3   3   3   3   -4   -4   -4

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {y}, {x});
    EXPECT_EQ((vector<output_c_type>{2, 2, 2, -3, -3, -3, 3, 3, 3, -4, -4, -4}),
              read_vector<output_c_type>(y));
}

NGRAPH_TEST(${BACKEND_NAME}, shape_of_scalar)
{
    Shape input_shape{};
    Shape output_shape{0};

    auto A = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto f = std::make_shared<Function>(std::make_shared<op::ShapeOf>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, input_shape);
    copy_data(a, vector<float>{0});
    auto result = backend->create_tensor(element::u64, output_shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<uint64_t> expected{};
    EXPECT_EQ(expected, read_vector<uint64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, shape_of_vector)
{
    Shape input_shape{2};
    Shape output_shape{1};

    auto A = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto f = std::make_shared<Function>(std::make_shared<op::ShapeOf>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, input_shape);
    copy_data(a, vector<float>(2, 0));
    auto result = backend->create_tensor(element::u64, output_shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<uint64_t> expected{2};
    EXPECT_EQ(expected, read_vector<uint64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, shape_of_matrix)
{
    Shape input_shape{2, 4};
    Shape output_shape{2};

    auto A = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto f = std::make_shared<Function>(std::make_shared<op::ShapeOf>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, input_shape);
    copy_data(a, vector<float>(2 * 4, 0));
    auto result = backend->create_tensor(element::u64, output_shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<uint64_t> expected{2, 4};
    EXPECT_EQ(expected, read_vector<uint64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, shape_of_5d)
{
    Shape input_shape{2, 4, 8, 16, 32};
    Shape output_shape{5};

    auto A = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto f = std::make_shared<Function>(std::make_shared<op::ShapeOf>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, input_shape);
    copy_data(a, vector<float>(2 * 4 * 8 * 16 * 32, 0));
    auto result = backend->create_tensor(element::u64, output_shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    vector<uint64_t> expected{2, 4, 8, 16, 32};
    EXPECT_EQ(expected, read_vector<uint64_t>(result));
}
