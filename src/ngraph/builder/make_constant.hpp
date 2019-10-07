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

#include "ngraph/node.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace builder
    {
        template <class T>
        std::shared_ptr<Node>
            make_constant(const element::Type& type, const Shape& shape, const T& num)
        {
            std::shared_ptr<Node> val = nullptr;

            if (type == element::f32)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<float>{static_cast<float>(num)});
            }
            else if (type == element::f64)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<double>{static_cast<double>(num)});
            }
            else if (type == element::f16)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type,
                    ngraph::Shape{},
                    std::vector<ngraph::float16>{ngraph::float16(static_cast<float>(num))});
            }
            else if (type == element::i64)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<int64_t>{static_cast<int64_t>(num)});
            }
            else if (type == element::i32)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<int32_t>{static_cast<int32_t>(num)});
            }
            else if (type == element::i16)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<int16_t>{static_cast<int16_t>(num)});
            }
            else if (type == element::i8)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<int8_t>{static_cast<int8_t>(num)});
            }
            else if (type == element::u64)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<uint64_t>{static_cast<uint64_t>(num)});
            }
            else if (type == element::u32)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<uint32_t>{static_cast<uint32_t>(num)});
            }
            else if (type == element::u16)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<uint16_t>{static_cast<uint16_t>(num)});
            }
            else if (type == element::u8)
            {
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<uint8_t>{static_cast<uint8_t>(num)});
            }
            else
            {
                throw ngraph_error("make_constant: Unsupported element type");
            }

            if (shape.size() > 0)
            {
                ngraph::AxisSet axes;
                for (size_t i = 0; i < shape.size(); i++)
                {
                    axes.insert(i);
                }
                val = std::make_shared<ngraph::op::Broadcast>(val, shape, axes);
            }

            return val->add_provenance_group_members_above({});
        }
    }
}
