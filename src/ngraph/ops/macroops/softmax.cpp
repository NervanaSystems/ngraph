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

#include "ngraph/ops/macroops/softmax.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/function.hpp"
#include <algorithm>
#include <numeric>
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/util.hpp"
#include <string>

using namespace ngraph::op;


std::shared_ptr<::ngraph::Node> SoftMax::lower()
{
	if (m_arguments.size() != 1) 
	{
		throw ngraph_error("Wrong number of arguments");
	}

	auto arg = m_arguments.at(0);
	auto st = get_shape_et(m_arguments.at(0));

	//TODO: [nikolayk] implement a numerically stable-flavour w/ max
	auto exp = std::make_shared<op::Exp>(arg);
	auto sum = std::make_shared<Sum>(exp, AxisSet({m_reduction_axis}));
	std::shared_ptr<Node> broadcasted_sum = std::make_shared<Broadcast>(sum, st.shape, AxisSet({ m_reduction_axis }));
	return exp / broadcasted_sum;
}
