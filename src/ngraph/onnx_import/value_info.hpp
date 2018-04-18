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

#include <map>
#include <ostream>
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "node.hpp"
#include "onnx.pb.h"

namespace ngraph
{
    namespace onnx_import
    {
        class ValueInfo
        {
            onnx::ValueInfoProto m_value_info_proto;
            Graph* m_graph_prt;

            friend std::ostream& operator<<(std::ostream& os, const ValueInfo& wrapper);

        public:
            ValueInfo(const onnx::ValueInfoProto& proto, Graph* graph_ptr);

            const ngraph::Shape get_shape() const;

            const ngraph::element::Type get_element_type() const;
        };

        std::ostream& operator<<(std::ostream& os, const ValueInfo& wrapper);
    } // namespace onnx_import
} // namespace ngraph
