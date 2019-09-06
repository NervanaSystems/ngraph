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

#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

const string op::GenerateMask::type_name{"GenerateMask"};

#if 0
// Not supported until all transformers use nodes instead of attributes
op::GenerateMask::GenerateMask(const Output<Node>& training,
                               const Output<Node>& shape,
                               const Output<Node>& probability,
                               const Output<Node>& seed,
                               const Output<Node>& use_seed,
                               const element::Type& element_type)
    : Op({training, shape, probability, seed, use_seed})
    , m_element_type(element_type)
{
}
#endif

op::GenerateMask::GenerateMask(const Output<Node>& training,
                               const Shape& shape,
                               const element::Type& element_type,
                               uint64_t seed,
                               double prob,
                               bool use_seed)
    : Op({training})
    , m_element_type(element_type)
    , m_shape(shape)
    , m_use_seed(use_seed)
    , m_seed(seed)
    , m_probability(prob)
{
    set_argument(1, make_shared<op::Constant>(element::u64, Shape{shape.size()}, shape));
    set_argument(2,
                 make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{use_seed}));
    set_argument(3, make_shared<op::Constant>(element::u64, Shape{}, std::vector<uint64_t>{seed}));
    set_argument(4, make_shared<op::Constant>(element::f64, Shape{}, std::vector<double>{prob}));
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::GenerateMask::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<GenerateMask>(
        new_args.at(0), m_shape, m_element_type, m_seed, m_probability, m_use_seed);
}

void ngraph::op::GenerateMask::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(0).compatible(PartialShape{}),
                          "Training node should be a scalar flag indicating a mode");

    NODE_VALIDATION_CHECK(
        this, m_element_type.is_static(), "Output element type must not be dynamic.");

    set_output_type(0, m_element_type, m_shape);
}
