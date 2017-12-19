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

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/runtime/ngvm/call_frame.hpp"
#include "ngraph/runtime/ngvm/parameterized_tensor_view.hpp"
#include "ngraph/runtime/ngvm/tensor_view_info.hpp"

namespace ngraph
{
    namespace runtime
    {
        class TensorViewInfo;

        namespace ngvm
        {
            class CallFrame;

            template <typename ET>
            typename ET::type* get_tensor_data_ptr(CallFrame& call_frame,
                                                   const TensorViewInfo& tensor_view_info)
            {
                return call_frame.get_tensor_view_data<ET>(tensor_view_info.get_index());
            }

            size_t get_tensor_element_count(CallFrame& call_frame,
                                            const TensorViewInfo& tensor_view_info)
            {
                return tensor_view_info
                    .get_layout<ngraph::descriptor::layout::DenseTensorViewLayout>()
                    ->get_size();
            }

            /// @brief Framework constructor of a tensor of a specific element type and shape.
            template <typename ET>
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>
                make_tensor(const Shape& shape)
            {
                return std::make_shared<runtime::ParameterizedTensorView<ET>>(shape);
            }

            /// @brief Framework constructor of a tensor of a specific element type and shape.
            template <typename ET>
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>
                make_tensor(const Shape& shape, const std::vector<typename ET::type>& data)
            {
                auto rc = std::make_shared<runtime::ParameterizedTensorView<ET>>(shape);
                rc->write(data.data(), 0, data.size() * sizeof(typename ET::type));
                return rc;
            }
        }
    }
}
