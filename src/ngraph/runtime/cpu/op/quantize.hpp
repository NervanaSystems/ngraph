
/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        class Quantize : public Op
        {
        public:
            Quantize(std::shared_ptr<Node> input,
                     std::shared_ptr<Node> min,
                     std::shared_ptr<Node> max,
                     const element::Type& type);
            const element::Type& get_quantize_et() const { return m_element_type; }
            float get_input_min() const { return m_input_min; }
            float get_input_max() const { return m_input_max; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            const element::Type m_element_type;
            float m_input_min;
            float m_input_max;
        };
    }
}
