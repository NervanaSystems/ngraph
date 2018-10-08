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

#include <onnx.pb.h>
#include <ostream>
#include <assertion.hpp>

#include "model.hpp"
#include "ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
            Model::Model(const onnx::ModelProto& model_proto)
                    : m_model_proto{&model_proto}
            {
                // Verify that the ONNX graph contains only nodes of supported op_type
                assert_all_op_types_supported();
            }

            void Model::assert_all_op_types_supported()
            {
                std::set<std::string> unsupported_ops;
                for (const auto& node_proto : get_graph().node())
                {
                    std::string op_type = node_proto.op_type();
                    if (!ops_bridge::is_op_type_supported(op_type))
                    {
                        unsupported_ops.insert(op_type);
                    }
                }

                std::string unsupported_ops_str;
                std::size_t index = 0;
                for(const auto& op_type: unsupported_ops) {
                    unsupported_ops_str += (index++ != 0 ? ", " : "");
                    unsupported_ops_str += op_type;
                }
                NGRAPH_ASSERT(unsupported_ops.empty()) << "unknown operations: " << unsupported_ops_str;
            }

    } // namespace onnx_import

} // namespace ngraph
