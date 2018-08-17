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

#include <ostream>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter_vector.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

#include <onnx.pb.h>

#include "node.hpp"
#include "tensor.hpp"

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
                        : ngraph_error{"value info has not element type specified"}
                    {
                    }
                };
                struct unsupported_element_type : ngraph_error
                {
                    explicit unsupported_element_type(onnx::TensorProto_DataType type)
                        : ngraph_error{"unsupported value info element type: " +
                                       onnx::TensorProto_DataType_Name(type)}
                    {
                    }
                };
            }
        }

        class ValueInfo
        {
        public:
            ValueInfo(ValueInfo&&) = default;
            ValueInfo(const ValueInfo&) = default;

            ValueInfo() = delete;
            explicit ValueInfo(const onnx::ValueInfoProto& value_info_proto)
                : m_value_info_proto{value_info_proto}
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

            const std::string& get_name() const { return m_value_info_proto.name(); }
            const Shape& get_shape() const { return m_shape; }
            const element::Type& get_element_type() const
            {
                if (!m_value_info_proto.type().tensor_type().has_elem_type())
                {
                    throw error::value_info::unspecified_element_type{};
                }
                switch (m_value_info_proto.type().tensor_type().elem_type())
                {
                case onnx::TensorProto_DataType::TensorProto_DataType_BOOL: return element::boolean;
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16: return element::f32;
                case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE: return element::f64;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT8: return element::i8;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT16: return element::i16;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT32: return element::i32;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT64: return element::i64;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT8: return element::u8;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT16: return element::u16;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT32: return element::u32;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT64: return element::u64;
                default:
                    throw error::value_info::unsupported_element_type{
                        m_value_info_proto.type().tensor_type().elem_type()};
                }
            }

            std::shared_ptr<ngraph::Node>
                get_ng_node(op::ParameterVector& parameters,
                            const std::map<std::string, Tensor>& initializers) const
            {
                const auto it{initializers.find(get_name())};
                if (it != std::end(initializers))
                {
                    return get_ng_constant(it->second);
                }
                else
                {
                    parameters.push_back(get_ng_parameter());
                    return parameters.back();
                }
            }

        protected:
            std::shared_ptr<op::Parameter> get_ng_parameter() const
            {
                return std::make_shared<op::Parameter>(get_element_type(), get_shape());
            }

            std::shared_ptr<op::Constant> get_ng_constant(const Tensor& tensor) const
            {
                switch (m_value_info_proto.type().tensor_type().elem_type())
                {
                case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
                    return make_ng_constant<bool>(element::boolean, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
                    return make_ng_constant<float>(element::f32, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
                    return make_ng_constant<double>(element::f64, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
                    return make_ng_constant<int8_t>(element::i8, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
                    return make_ng_constant<int16_t>(element::i16, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
                    return make_ng_constant<int32_t>(element::i32, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
                    return make_ng_constant<int64_t>(element::i64, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
                    return make_ng_constant<uint8_t>(element::u8, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
                    return make_ng_constant<uint16_t>(element::u16, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
                    return make_ng_constant<uint32_t>(element::u32, tensor);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
                    return make_ng_constant<uint64_t>(element::u64, tensor);
                default:
                    throw error::value_info::unsupported_element_type{
                        m_value_info_proto.type().tensor_type().elem_type()};
                }
            }

            template <typename T>
            std::shared_ptr<op::Constant> make_ng_constant(const element::Type& type,
                                                           const Tensor& tensor) const
            {
                return std::make_shared<op::Constant>(type, m_shape, tensor.get_data<T>());
            }

        private:
            const onnx::ValueInfoProto& m_value_info_proto;
            Shape m_shape;
        };

        inline std::ostream& operator<<(std::ostream& outs, const ValueInfo& info)
        {
            return (outs << "<ValueInfo: " << info.get_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
