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

namespace ngraph
{
    class BroadcastOp : public BuiltinOp
    {
    public:
        ///
        /// @param arg The tensor view to be broadcast.
        /// @param shape The shape of the result
        /// @param broadcast_axes The axis positions (0-based) in the result that are being broadcast.
        ///  the remaining axes in shape must be the same as the shape of arg.
        ///
        BroadcastOp(const std::shared_ptr<Node>& arg, const Shape& shape, const AxisSet& broadcast_axes)
            : BuiltinOp({arg})
            , m_shape(shape)
            , m_broadcast_axes(broadcast_axes)
        {
        }

        virtual std::string get_op_class_name() const override { return "broadcast"; }
        virtual void        propagate_types() override;

    protected:
        Shape               m_shape;
        AxisSet m_broadcast_axes;
    };

    namespace op
    {
        std::shared_ptr<Node> broadcast(const std::shared_ptr<Node>&           tensor,
                            const Shape&               shape,
                            AxisSet&& broadcast_axes);
    }
}
