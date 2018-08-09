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
#include <sstream>

#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/ngraph.hpp"
#include "util/test_tools.hpp"

TEST(onnx, model_add_abc)
{
    auto model{ngraph::onnx_import::load_onnx_model(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"))};
    auto backend{ngraph::runtime::Backend::create("CPU")};

    ngraph::Shape shape{1};
    auto a{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(a, std::vector<float>{1});
    auto b{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(b, std::vector<float>{2});
    auto c{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(c, std::vector<float>{3});

    auto r{backend->create_tensor(ngraph::element::f32, shape)};

    backend->call(model.front(), {r}, {a, b, c});
    EXPECT_EQ((std::vector<float>{6}), read_vector<float>(r));
}

TEST(onnx, model_add_abc_initializers)
{
    auto model{ngraph::onnx_import::load_onnx_model(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.onnx"))};
    auto backend{ngraph::runtime::Backend::create("CPU")};

    ngraph::Shape shape{2, 2};

    auto c{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(c, std::vector<float>{1, 2, 3, 4});

    auto r{backend->create_tensor(ngraph::element::f32, shape)};

    backend->call(model.front(), {r}, {c});
    EXPECT_EQ((std::vector<float>{3, 6, 9, 12}), read_vector<float>(r));
}

TEST(onnx, model_split_default)
{
    auto function{ngraph::onnx_import::import_onnx_function(
            ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/split_default.onnx"))};
    auto backend{ngraph::runtime::Backend::create("CPU")};
}

TEST(onnx, model_split)
{
    auto function{ngraph::onnx_import::import_onnx_function(
            ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/split.onnx"))};
    auto backend{ngraph::runtime::Backend::create("CPU")};
}

TEST(onnx, model_batchnorm_default)
{
    // Batch Normalization with default parameters
    auto function{ngraph::onnx_import::import_onnx_function(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.onnx"))};


    std::vector<std::vector<float>> args;

    // input data shape: (2, 3, 4, 5)
    args.emplace_back{ngraph::test::NDArray<float, 4>(
        {{{{-0.7685804, 0.02295618, -0.713669, 1.5440737, 0.07175534},
           { 0.9465769, -1.1485962, -1.6772385, 0.8595791, -3.0079234},
           { 0.83140105, 1.1823093, 1.3203293, -0.3026504, 1.2072734},
           { 1.0135608, -0.42294255, 1.195275 , -0.39429635, -1.0982096}},

          {{ 0.94553566, 0.04512986, -1.0354314, -1.4168428, 0.6599153},
           { 1.118901,  -1.1888205, -2.636044 , 1.2197105, -0.5531194},
           {-0.18257885, 0.36262384, -1.9582844, 2.2532516, -2.1224964},
           {-0.5528226, 0.07029665, -1.0828252, 1.8929064, 0.1968072}},

          {{-1.4852247, 0.8238424, -1.8925015, -0.04180111, 0.5097965},
           { 1.0227106, 0.08579426, 0.24652481, -0.32846594, 0.1779821},
           { 0.02376763, -0.5985626, -0.5902406, 0.48650023, -0.7529659},
           { 1.5400531, 2.35896  , -0.00402553, -1.9918069, -1.1199996}}},

         {{{ 0.7363347, -1.1609254, 0.7937933, 2.117957 , 0.4851638},
           { 0.5204418, 0.61928153, -0.32841128, 0.6854336, 0.5557987},
           {-0.09866326, -1.069918 , 0.6336388, 0.49344212, 0.20250301},
           {-1.0538961, -1.1171275, -1.5483406, 0.61678165, 1.5143661}},

          {{ 0.12790644, -0.4659769, 0.97132444, 0.2983505, -0.06221787},
           { 2.1492324, -0.45414695, -0.04797961, 0.58100474, -0.9931606},
           {-0.05125588, -1.3939717, -0.24935935, -0.4224357, -0.27554026},
           {-0.5127014, -1.8032415, -0.38176435, 1.8393202, 1.3186238}},

          {{ 0.41312563, 0.18794955, 0.83369243, 2.5602016, -0.15156993},
           { 0.79013413, 1.2272907, -0.06972152, 0.8923214, 0.21695873},
           {-0.8759175, -0.80731165, 0.85726297, 0.5651181, -1.4697937},
           { 0.31026196, -1.7356321, 1.4899335, -0.6872855, -0.99874365}}}}).get_vector()};
       
    // scale (3)
    args.emplace_back{std::vector<float>{-1.7599288, -1.1512759, 0.64616394}};
    // bias (3)
    args.emplace_back{std::vector<float>{-0.40034863, 0.64600074, -1.3011926}};
    // mean (3)
    args.emplace_back{std::vector<float>{0.8868739, -0.9037939, -1.326596}};
    // var (3)
    args.emplace_back{std::vector<float>{0.48378697, 0.50140697, 0.25334656}};

    // shape (2, 3, 4, 5)
    auto expected_output = 
        ngraph::test::NDArray<float, 4>(
        {{{{ 3.7883654 ,  1.7855797 ,  3.649426  , -2.0632288 ,  1.6621056},
           {-0.55141217,  4.7499003 ,  6.087498  , -0.33128592,  9.454465 },
           {-0.25998843, -1.1478741 , -1.4970994 ,  2.609446  , -1.2110395},
           {-0.72089815,  2.9138155 , -1.1806805 ,  2.841333  ,  4.6224103}},

          {{-2.3607278 , -0.8968047 ,  0.8600233 ,  1.4801402 , -1.8963526 },
           {-2.6425934 ,  1.1094106 ,  3.462376  , -2.8064942 ,  0.07585746},
           {-0.5265851 , -1.4130019 ,  2.360442  , -4.4868746 ,  2.6274257 },
           { 0.07537484, -0.9377223 ,  0.9370784 , -3.9010081 , -1.1434091 }},

          {{-1.5048305 ,  1.4594082 , -2.0276675 ,  0.34814835,  1.0562555 },
           { 1.7147032 ,  0.5119474 ,  0.7182833 , -0.01985431,  0.63029253},
           { 0.43232143, -0.36658794, -0.35590464,  1.0263492 , -0.56480145},
           { 2.3788357 ,  3.430098  ,  0.3966422 , -2.1551495 , -1.0359769 }}},

         {{{-0.01944667,  4.781097  , -0.16483143, -3.5152977 ,  0.6160785 },
           { 0.5268165 ,  0.2767271 ,  2.6746273 ,  0.10934576,  0.4373546 },
           { 2.0933073 ,  4.550825  ,  0.24039963,  0.59513235,  1.3312812 },
           { 4.510286  ,  4.670277  ,  5.7613544 ,  0.2830524 , -1.9880613 }},

          {{-1.0313871 , -0.06582302, -2.402656  , -1.3085032 , -0.7222737 },
           {-4.317755  , -0.08505672, -0.7454231 , -1.768056  ,  0.79129744},
           {-0.74009633,  1.4429553 , -0.41801023, -0.13661438, -0.37544394},
           { 0.01014388,  2.1083655 , -0.20273983, -3.8138852 , -2.9673119 }},

          {{ 0.9321555 ,  0.6430881 ,  1.4720529 ,  3.6884398 ,  0.20723379},
           { 1.4161359 ,  1.9773308 ,  0.31230593,  1.5473174 ,  0.68032837},
           {-0.7226392 , -0.6345672 ,  1.5023116 ,  1.1272739 , -1.485021  },
           { 0.8001052 , -1.826288  ,  2.3144956 , -0.48048496, -0.8803159 }}}}).get_vector();

    // std::vector<std::vector<float>> inputs;
    // inputs.emplace_back{read_tensor_proto_data_file<float>(
    //     ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default_input_0.pb"))};

    // auto expected_output = read_tensor_proto_data_file<float>(
    //     ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default_output_0.pb"));

    auto result_vectors = execute(function, inputs, "CPU");
    EXPECT_EQ(expected_output, result_vectors.front());
}