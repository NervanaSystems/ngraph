//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <memory> // std::shared_ptr, std::make_shared
#include <onnx/onnx_pb.h>
#include <vector>

#include "core/node.hpp"
#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace utils
        {
            ///
            /// \brief      Return a vector of nGraph nodes obtained by expanding the ONNX fuction.
            ///
            /// \param[in]  node                The node representing incoming ONNX operation.
            ///
            /// \return     Vector of nGraph nodes equivalent of the ONNX operation.
            ///
            NodeVector expand_onnx_function(const Node& node);

            ONNX_NAMESPACE::TypeProto get_proto_type(element::Type type, Shape shape);

            std::vector<std::shared_ptr<ngraph::Node>>
                get_expanded_function(ONNX_NAMESPACE::NodeProto* new_node,
                                      ONNX_NAMESPACE::GraphProto graph,
                                      int opset_version);
        }
    }
}
