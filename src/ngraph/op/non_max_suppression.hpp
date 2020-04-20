//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise addition operation.
            ///
            class NGRAPH_API NonMaxSuppression : public Op
            {
            public:
                enum class BoxEncodingType
                {
                    CORNER,
                    CENTER
                };

                static constexpr NodeTypeInfo type_info{"NonMaxSuppression", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                NonMaxSuppression() = default;

                /// \brief Constructs a NonMaxSuppression operation.
                ///
                /// \param boxes Output that produces a tensor with box coordinates.
                /// \param scores Output that produces ta tensor
                /// \param max_output_boxes_per_class Auto broadcast specification
                /// \param iou_threshold Auto broadcast specification
                /// \param score_threshold Auto broadcast specification
                /// \param box_encoding Auto broadcast specification
                /// \param sort_result_descending Auto broadcast specification
                /// \param output_type TODO
                NonMaxSuppression(const Output<Node>& boxes,
                                  const Output<Node>& scores,
                                  const Output<Node>& max_output_boxes_per_class,
                                  const Output<Node>& iou_threshold,
                                  const Output<Node>& score_threshold,
                                  const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                                  const bool sort_result_descending = true,
                                  const ngraph::element::Type& output_type = ngraph::element::i64);

                /// \brief Constructs a NonMaxSuppression operation with default values for the last
                ///        3 inputs
                ///
                /// \param boxes Output that produces a tensor with box coordinates.
                /// \param scores Output that produces ta tensor
                /// \param box_encoding Auto broadcast specification
                /// \param sort_result_descending Auto broadcast specification
                /// \param output_type TODO
                NonMaxSuppression(const Output<Node>& boxes,
                                  const Output<Node>& scores,
                                  const BoxEncodingType box_encoding = BoxEncodingType::CORNER,
                                  const bool sort_result_descending = true,
                                  const ngraph::element::Type& output_type = ngraph::element::i64);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                BoxEncodingType get_box_encoding() const { return m_box_encoding; }
                void set_box_encoding(const BoxEncodingType box_encoding)
                {
                    m_box_encoding = box_encoding;
                }
                bool get_sort_result_descending() const { return m_sort_result_descending; }
                void set_sort_result_descending(const bool sort_result_descending)
                {
                    m_sort_result_descending = sort_result_descending;
                }

                element::Type get_output_type() const { return m_output_type; }
                void set_output_type(const element::Type& output_type)
                {
                    m_output_type = output_type;
                }
                using Node::set_output_type;

            protected:
                BoxEncodingType m_box_encoding = BoxEncodingType::CORNER;
                bool m_sort_result_descending = true;
                ngraph::element::Type m_output_type = ngraph::element::i64;

            private:
                int64_t max_boxes_output_from_input() const;
            };
        }
    }

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s,
                             const op::v1::NonMaxSuppression::BoxEncodingType& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v1::NonMaxSuppression::BoxEncodingType>
        : public EnumAttributeAdapterBase<op::v1::NonMaxSuppression::BoxEncodingType>
    {
    public:
        AttributeAdapter(op::v1::NonMaxSuppression::BoxEncodingType& value)
            : EnumAttributeAdapterBase<op::v1::NonMaxSuppression::BoxEncodingType>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v1::NonMaxSuppression::BoxEncodingType>", 1};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
}
