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

#include "ngraph/op/util/op_annotations.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            /// \brief Annotations added to graph ops by CPU backend passes
            class CPUOpAnnotations : public ngraph::op::util::OpAnnotations
            {
            public:
                CPUOpAnnotations() {}
                bool is_mkldnn_op() { return m_mkldnn_op; }
                void set_mkldnn_op(bool val) { m_mkldnn_op = val; }
            private:
                bool m_mkldnn_op = false;
            };
        }
    }
}
