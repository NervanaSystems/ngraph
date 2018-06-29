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

#include <vector>
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "node.hpp"
#include "onnx.pb.h"

namespace ngraph
{
    namespace onnx_import
    {
        class Tensor
        {
            onnx::TensorProto m_tensor_proto;
            const Graph* m_graph_ptr;

            friend std::ostream& operator<<(std::ostream& os, const Tensor& wrapper);

        public:
            Tensor(const onnx::TensorProto& proto, const Graph* graph_ptr);

            template <typename T>
            std::vector<T> get_vector() const;
        };

        std::ostream& operator<<(std::ostream&, const Tensor&);

    } // namespace onnx_import
} // namespace ngraph
