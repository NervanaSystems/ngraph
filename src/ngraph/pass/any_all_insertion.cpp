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

#include "any_all_insertion.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

static bool is_boolean_scalar_constant_with_val(std::shared_ptr<ngraph::Node> node, bool val)
{
    auto k = std::dynamic_pointer_cast<op::Constant>(node);

    if (k == nullptr)
    {
        return false;
    }

    if (k->get_element_type() != element::boolean)
    {
        return false;
    }

    if (k->get_shape() != Shape{})
    {
        return false;
    }

    const char* k_data = k->get_data_ptr<char>();
    return (*k_data == static_cast<char>(val));
}

template <typename T>
static bool check_reduce_for_replacement(std::shared_ptr<ngraph::op::Reduce> reduce,
                                         bool expected_k_val)
{
    auto reductee = reduce->get_argument(0);
    auto init_val = reduce->get_argument(1);

    if (!is_boolean_scalar_constant_with_val(init_val, expected_k_val))
    {
        return false;
    }

    auto func = reduce->get_functions().at(0);
    auto func_result_op = func->get_results().at(0)->get_argument(0);

    if (std::dynamic_pointer_cast<T>(func_result_op) == nullptr)
    {
        return false;
    }

    auto func_params = func->get_parameters();
    auto func_param_0 = func_params.at(0);
    auto func_param_1 = func_params.at(1);
    auto func_result_op_arg_0 = func_result_op->get_argument(0);
    auto func_result_op_arg_1 = func_result_op->get_argument(1);

    if (!((func_param_0 == func_result_op_arg_0 && func_param_1 == func_result_op_arg_1) ||
          (func_param_0 == func_result_op_arg_1 && func_param_1 == func_result_op_arg_0)))
    {
        return false;
    }

    return true;
}

bool ngraph::pass::AnyAllInsertion::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    auto reduce = std::dynamic_pointer_cast<ngraph::op::Reduce>(node);
    if (reduce == nullptr)
    {
        return false;
    }

    if (check_reduce_for_replacement<op::Or>(reduce, false))
    {
        ngraph::replace_node(
            reduce,
            std::make_shared<op::Any>(reduce->get_argument(0), reduce->get_reduction_axes()));
        return true;
    }
    else if (check_reduce_for_replacement<op::And>(reduce, true))
    {
        ngraph::replace_node(
            reduce,
            std::make_shared<op::All>(reduce->get_argument(0), reduce->get_reduction_axes()));
        return true;
    }

    return false;
}
