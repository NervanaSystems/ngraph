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

#include "ngraph/common.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/ndarray.hpp"

namespace ngraph
{
    namespace element
    {
        class Type;
    }

    namespace runtime
    {
        class ExternalFunction;
        class CallFrame;
        class TensorView;
        class Tuple;
        class Value;

        template <typename ET>
        class ParameterizedTensorView;

        /// @brief Interface to a generic backend.
        ///
        /// Backends are responsible for function execution and value allocation.
        class Backend
        {
        public:
            virtual ~Backend() {}
            /// @brief Make a call frame that can support one concurrent call of an external function.
            ///
            /// If more than one concurrent execution is needed, each execution will require its own call frame.
            virtual std::shared_ptr<ngraph::runtime::CallFrame>
                make_call_frame(const std::shared_ptr<ExternalFunction>& external_function) = 0;

            /// @brief Return a handle for a tensor on the backend device.
            virtual std::shared_ptr<ngraph::runtime::TensorView>
                make_primary_tensor_view(const ngraph::element::Type& element_type,
                                         const Shape& shape);

            template <typename ET>
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>
                make_parameterized_tensor_view(const Shape& shape)
            {
                return std::dynamic_pointer_cast<ngraph::runtime::ParameterizedTensorView<ET>>(
                    make_primary_tensor_view(ET::element_type(), shape));
            }

            template <typename ET>
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>
                make_parameterized_tensor_view(const NDArrayBase<typename ET::type>& ndarray)
            {
                auto result =
                    std::dynamic_pointer_cast<ngraph::runtime::ParameterizedTensorView<ET>>(
                        make_primary_tensor_view(ET::element_type(), ndarray.get_shape()));
                *result = ndarray;
                return result;
            }

            /// @brief Construct a tuple handle from a sequence of values.
            virtual std::shared_ptr<ngraph::runtime::Tuple>
                make_tuple(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& elements);
        };
    }
}
