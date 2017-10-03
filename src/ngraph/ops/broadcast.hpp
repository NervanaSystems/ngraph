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
    namespace op
    {
        class Broadcast : public IndexBuiltin
        {
        public:
            ///
            /// @param arg The tensor view to be broadcast.
            /// @param shape The shape of the result
            /// @param broadcast_axes The axis positions (0-based) in the result that are being broadcast.
            ///  the remaining axes in shape must be the same as the shape of arg.
            ///
            Broadcast(const std::shared_ptr<Node>& arg,
                      const Shape&                 shape,
                      const AxisSet&               broadcast_axes)
                : IndexBuiltin(arg)
                , m_shape(shape)
                , m_broadcast_axes(broadcast_axes)
            {
            }

            virtual std::string description() const override { return "Broadcast"; }
            virtual void        propagate_types() override;

            const AxisSet& get_broadcast_axes() const { return m_broadcast_axes; }

        protected:
            Shape   m_shape;
            AxisSet m_broadcast_axes;
        };
    }
}
