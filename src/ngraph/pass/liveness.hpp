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

#pragma once

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class Liveness;
    }
}

/**
 * \brief Determine the first and last operations in the specified function that require each
 * tensor.
 *
 * This pass assumes that the function's 'get_ordered_ops()' method describes the execution order
 * for the functions ops.
 *
 * This pass updates each of those ops as follows (except as described further down):
 * - The op's 'liveness_new_list' contains pointers to all tensors that are created by the
 *   execution of this op.
 * - The op's 'liveness_free_list' contains pointers to all tensors for which this op is their final
 *   use.
 * - If an op outputs a tensor that has no users, a pointer to that tensor appears in both the
 *   'liveness_new_list' *and* 'liveness_free_list' of that op.
 *
 * The following tensors will not be added to any of the function's ops' liveness-new/free lists:
 * - Tensors that are populated by ngraph Constant ops.
 * - Tensors that provide the function's input parameter values.
 * - Tensors that store the function's result values.
 */
class ngraph::pass::Liveness : public FunctionPass
{
public:
    bool run_on_function(std::shared_ptr<ngraph::Function>) override;
};
