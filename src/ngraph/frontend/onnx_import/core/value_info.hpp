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

#include <onnx-ml.pb.h>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "node.hpp"
#include "tensor.hpp"
#include "utils/common.hpp"
#include "weight.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            namespace value_info
            {
                struct unspecified_element_type : ngraph_error
                {
                    unspecified_element_type()
                        : ngraph_error{"value info has no element type specified"}
                    {
                    }
                };
            } // namespace value_info
        }     // namespace error

        class ValueInfo
        {
        public:
            ValueInfo(ValueInfo&&) = default;
            ValueInfo(const ValueInfo&) = default;

            ValueInfo() = delete;
            explicit ValueInfo(const onnx::ValueInfoProto& value_info_proto)
                : m_value_info_proto{&value_info_proto}
            {
                if (value_info_proto.type().has_tensor_type())
                {
                    for (const auto& dim : value_info_proto.type().tensor_type().shape().dim())
                    {
                        m_shape.emplace_back(static_cast<Shape::value_type>(dim.dim_value()));
                    }
                }
            }

            ValueInfo& operator=(const ValueInfo&) = delete;
            ValueInfo& operator=(ValueInfo&&) = delete;

            const std::string& get_name() const { return m_value_info_proto->name(); }
            const Shape& get_shape() const { return m_shape; }
            const element::Type& get_element_type() const
            {
                if (!m_value_info_proto->type().tensor_type().has_elem_type())
                {
                    throw error::value_info::unspecified_element_type{};
                }
                return common::get_ngraph_element_type(
                    m_value_info_proto->type().tensor_type().elem_type());
            }

            std::shared_ptr<ngraph::Node>
                get_ng_node(ParameterVector& parameters,
                            const std::map<std::string, Tensor>& initializers,
                            const Weights& weights = {}) const
            {
                const auto it = initializers.find(get_name());
                if (it != std::end(initializers))
                {
                    return get_ng_constant(it->second);
                }
                else
                {
                    const auto pt = weights.find(get_name());
                    if (pt != std::end(weights))
                    {
                        return get_ng_constant(pt->second);
                    }
                }
                parameters.push_back(get_ng_parameter());
                return parameters.back();
            }

        protected:
            std::shared_ptr<op::Parameter> get_ng_parameter() const
            {
                return std::make_shared<op::Parameter>(get_element_type(), get_shape());
            }

            std::shared_ptr<op::Constant> get_ng_constant(const Weight& weight) const
            {
                return std::make_shared<op::Constant>(weight.type(), weight.shape(), weight.data());
            }

            std::shared_ptr<op::Constant> get_ng_constant(const Tensor& tensor) const
            {
                return tensor.get_ng_constant();
            }

        private:
            const onnx::ValueInfoProto* m_value_info_proto;
            Shape m_shape;
        };

        inline std::ostream& operator<<(std::ostream& outs, const ValueInfo& info)
        {
            return (outs << "<ValueInfo: " << info.get_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
