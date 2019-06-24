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

#pragma once
#include <memory>
#include <unordered_map>

#include "ngraph/pass/pass.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class Concat;
        class Dequantize;
        class Pad;
        class Quantize;
        class Reshape;
        class Slice;
        namespace util
        {
            class UnaryElementwiseArithmetic;
            class BinaryElementwiseArithmetic;
        }
    }
    namespace pass
    {
        using ReshapeMap = std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<op::Reshape>>;
        class ReshapeSinking : public ngraph::pass::FunctionPass
        {
        public:
            ReshapeSinking() { set_property(PassProperty::REQUIRE_STATIC_SHAPE, true); }
            bool run_on_function(std::shared_ptr<ngraph::Function> function) override;

            void swim(Input<Node> input, std::shared_ptr<op::Reshape> reshape);
            void sink_reshape(std::shared_ptr<op::Reshape> reshape);
            void sink_unary(std::shared_ptr<op::util::UnaryElementwiseArithmetic> n);
            void sink_binary(std::shared_ptr<op::util::BinaryElementwiseArithmetic> binary);
            void sink_slice(std::shared_ptr<op::Slice> n);
            void sink_pad(std::shared_ptr<op::Pad> n);
            void sink_quantize(std::shared_ptr<op::Quantize> quantize);
            void sink_concat(std::shared_ptr<op::Concat> n);
            void sink_dequantize(std::shared_ptr<op::Dequantize> dequantize);

        private:
            void materialize_shapes(std::shared_ptr<Node> n);
            void mark_reshape_for_deletion(std::shared_ptr<Node> reshape);
            void convert_binary_to_default_order(std::shared_ptr<Node> binary,
                                                 const Input<Node>& input,
                                                 std::shared_ptr<Node> right);

            std::deque<std::shared_ptr<Node>> m_nodes;
            ReshapeMap reorders;
            std::set<std::shared_ptr<Node>> reshapes_to_delete;
        };
    }
}

extern template ngraph::AxisVector
    ngraph::apply_permutation<ngraph::AxisVector>(ngraph::AxisVector input,
                                                  ngraph::AxisVector order);

extern template ngraph::Coordinate
    ngraph::apply_permutation<ngraph::Coordinate>(ngraph::Coordinate input,
                                                  ngraph::AxisVector order);

extern template ngraph::Strides
    ngraph::apply_permutation<ngraph::Strides>(ngraph::Strides input, ngraph::AxisVector order);

extern template ngraph::Shape ngraph::apply_permutation<ngraph::Shape>(ngraph::Shape input,
                                                                       ngraph::AxisVector order);
