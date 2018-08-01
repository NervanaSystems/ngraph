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

#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class Dequantize : public util::RequiresTensorViewArgs
        {
        public:
            Dequantize(std::shared_ptr<Node> input,
                       const float input_min_range,
                       const float input_max_range,
                       const element::Type& type);

            const float get_input_min() const { return m_input_min; }
            const float get_input_max() const { return m_input_max; }
            //TODO:Templatize it.
            const element::Type& get_dequantize_et() const { return m_element_type; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            const float m_input_min;
            const float m_input_max;
            const element::Type m_element_type;
        };
    }
}
