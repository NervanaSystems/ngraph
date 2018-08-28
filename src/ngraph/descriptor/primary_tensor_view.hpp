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

#include <string>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/descriptor/tensor_view.hpp"

namespace ngraph
{
    class Node;

    namespace descriptor
    {
        /// @brief A PrimaryTensorView owns the tensor. All other views are the result
        /// of some index operation on the primary view.
        class PrimaryTensorView : public TensorView
        {
        public:
            /// @param tensor_view_type The type for this view.
            /// @param name Description of the tensor, for debugging.
            PrimaryTensorView(const std::shared_ptr<const TensorViewType>& tensor_view_type,
                              const std::string& name);

            virtual const Tensor& get_tensor() const override;
            virtual Tensor& get_tensor() override;
            void set_tensor_view_type(const element::Type& element_type,
                                      const Shape& shape) override;

        protected:
            Tensor m_tensor;
        };
    }
}
