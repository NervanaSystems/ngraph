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

#include "ngraph/runtime/cpu/op/dropout.hpp"

#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Dropout::type_info;

op::Dropout::Dropout(const Output<Node>& input,
                     const Output<Node>& gm_const,
                     const Output<Node>& use_seed,
                     const Output<Node>& seed,
                     const Output<Node>& keep_prob)
    : Op({input, gm_const, use_seed, seed, keep_prob})
{
    constructor_validate_and_infer_types();

    set_output_size(2);
    set_output_type(0, get_input_element_type(0), input.get_shape());
    set_output_type(1, get_input_element_type(0), input.get_shape());
}

shared_ptr<Node> op::Dropout::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 5)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<Dropout>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
}

bool op::Dropout::get_use_seed() const
{
    bool use_seed = false;
    if (auto const_op =
            as_type_ptr<op::Constant>(input(2).get_source_output().get_node_shared_ptr()))
    {
        auto use_seed_ptr = static_cast<const int32_t*>(const_op->get_data_ptr());
        use_seed = static_cast<const bool>(*use_seed_ptr);
    }
    return use_seed;
}

uint64_t op::Dropout::get_seed() const
{
    uint64_t seed = 0;
    if (auto const_op =
            as_type_ptr<op::Constant>(input(3).get_source_output().get_node_shared_ptr()))
    {
        auto seed_ptr = static_cast<const uint64_t*>(const_op->get_data_ptr());
        seed = *seed_ptr;
    }
    return seed;
}
