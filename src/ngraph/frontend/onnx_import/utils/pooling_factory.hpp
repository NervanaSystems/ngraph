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
#include <type_traits>

#include "core/node.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace pooling
        {
            ///
            /// \brief      Factory class which generates sub-graphs for ONNX 'regular' pooling
            ///             operators.
            ///
            /// \note       This factory is intended for creating pooling operations like:
            ///             - AveragePool
            ///             - MaxPool
            ///
            ///             This base class holds all common attributes like srides, dilations,
            ///             paddings, kernel shape and auto_pad type.
            ///
            /// \see        GlobalPoolingFactory
            class PoolingFactory
            {
            public:
                explicit PoolingFactory(const Node& node);
                virtual ~PoolingFactory() = default;

                ///
                /// \brief      Creates a sub-graph representing appropriate ONNX operation.
                ///
                /// \tparam     NgraphOperator  nGraph operator class type used to build ONNX
                ///                             operation.
                ///
                /// \return     Vector of output nodes.
                ///
                template <typename NgraphOperator>
                NodeVector make_pooling_op() const
                {
                    return {std::make_shared<NgraphOperator>(m_inputs.at(0),
                                                             m_kernel_shape,
                                                             m_strides,
                                                             m_padding_below,
                                                             m_padding_above,
                                                             m_auto_pad)};
                }

            protected:
                Node m_onnx_node;
                const NodeVector m_inputs;
                Shape m_kernel_shape;
                Strides m_strides;
                Strides m_dilations;
                Shape m_padding_below;
                Shape m_padding_above;
                ngraph::op::PadType m_auto_pad;
            };

            // AvgPool accepts some additional parameters thus we have specialization for it.
            template <>
            NodeVector PoolingFactory::make_pooling_op<ngraph::op::AvgPool>() const;

            ///
            /// \brief      Factory class which generates sub-graphs for ONNX 'global' pooling
            ///             operators.
            ///
            class GlobalPoolingFactory : public PoolingFactory
            {
            public:
                explicit GlobalPoolingFactory(const Node& node);
                virtual ~GlobalPoolingFactory() = default;
            };

        } // namespace pooling
    }     // namespace onnx_import
} // namespace ngraph
