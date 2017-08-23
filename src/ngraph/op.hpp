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

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    class Op
    {
    };

    class Broadcast : public Op
    {
        class BroadcastCall : public Call
        {
            friend class Broadcast;

        public:
            BroadcastCall(const Node::ptr& arg, size_t axis)
                : Call({arg})
                , m_axis(axis)
            {
            }

            Op& op() const override;

        protected:
            size_t m_axis;
        };

    public:
        std::shared_ptr<BroadcastCall> operator()(const Node::ptr& tensor, size_t axis)
        {
            return std::make_shared<BroadcastCall>(tensor, axis);
        }
    };

    namespace op
    {
        extern Broadcast broadcast;
    }

    class Dot : public Op
    {
        class DotCall : public Call
        {
            friend class Dot;

        public:
            DotCall(const std::shared_ptr<Node>& arg0, const Node::ptr& arg1)
                : Call({arg0, arg1})
            {
            }

            Op& op() const override;
        };

    public:
        std::shared_ptr<DotCall> operator()(const Node::ptr& arg0, const Node::ptr& arg1)
        {
            return std::make_shared<DotCall>(arg0, arg1);
        }
    };

    namespace op
    {
        extern Dot dot;
    }
}
