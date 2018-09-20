//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <string>

#include "ngraph/node.hpp"
#include "ngraph/op/util/op_annotations.hpp"

namespace ngraph
{
    namespace op
    {
        /// Root of all actual ops
        class Op : public Node
        {
        public:
            void set_op_annotations(std::shared_ptr<ngraph::op::util::OpAnnotations> op_annotations)
            {
                m_op_annotations = op_annotations;
            }
            std::shared_ptr<ngraph::op::util::OpAnnotations> get_op_annotations() const
            {
                return m_op_annotations;
            }

        protected:
            Op(const std::string& node_type, const NodeVector& arguments);

        private:
            std::shared_ptr<ngraph::op::util::OpAnnotations> m_op_annotations;
        };
    }
}
