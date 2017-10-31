// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/ops/macroops/cross_entropy.hpp"
#include <algorithm>
#include <numeric>
#include <string>
#include "ngraph/function.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/macroops/mean.hpp"
#include "ngraph/ops/macroops/softmax.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/util.hpp"

using namespace ngraph::op;

std::shared_ptr<::ngraph::Node> CrossEntropy::lower()
{
    if (m_arguments.size() != 2)
    {
        throw ngraph_error("Wrong number of arguments");
    }

    auto predictions = m_arguments.at(0);
    auto answers = m_arguments.at(1);
    auto pred_st = get_shape_et(predictions);
    auto answers_st = get_shape_et(answers);

    if (answers_st.shape != pred_st.shape)
    {
        throw ngraph_error("Shapes don't match");
    }

    if (answers_st.type != pred_st.type)
    {
        throw ngraph_error("Types don't match");
    }

    //TODO [nikolayk] support vectors????
    if (answers_st.shape.size() != 2)
    {
        throw ngraph_error("Shape isn't supported");
    }

    auto logs = std::make_shared<op::Log>(answers);
    auto sum = std::make_shared<Sum>(predictions * logs, AxisSet({1}));
    auto neg = std::make_shared<op::Negative>(sum);
    auto mean = std::make_shared<Mean>(neg, AxisSet({0}));
    return mean;
}
