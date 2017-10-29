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

#pragma once

#include "ngraph/ops/op.hpp"
#include "ngraph/ops/macro.hpp"

namespace ngraph
{
	namespace op
	{
		class Mean : public MacroNode
		{
		public:
			Mean(const std::shared_ptr<Node>& arg, const AxisSet& reduction_axes)
				: MacroNode({ arg })
				, m_reduction_axes(reduction_axes)
			{
			}

			virtual std::shared_ptr<Node> lower() override;
			const AxisSet& get_reduction_axes() const { return m_reduction_axes; }
		protected:
			AxisSet m_reduction_axes;
		};
	}
}
