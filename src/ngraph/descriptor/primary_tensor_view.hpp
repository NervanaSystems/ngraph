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

#include <string>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/log.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/types/type.hpp"

namespace ngraph
{
    class Node;

    namespace descriptor
    {
        class Tensor;
        class TensorViewLayout;

        /// @brief A PrimaryTensorView owns the tensor. All other views are the result
        /// of some index operation on the primary view.
        class PrimaryTensorView : public TensorView
        {
        public:
            PrimaryTensorView(const std::shared_ptr<const TensorViewType>& tensor_view_type,
                              const std::string&                           name,
                              bool                                         is_output,
                              bool                                         is_input);

            virtual const Tensor& get_tensor() const override;
            virtual Tensor&       get_tensor() override;

        protected:
            Tensor m_tensor;
        };
    }
}
