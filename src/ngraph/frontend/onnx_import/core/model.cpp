//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <onnx-ml.pb.h>

#include "model.hpp"
#include "ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        Model::Model(const onnx::ModelProto& model_proto)
            : m_model_proto{&model_proto}
        {
            // Walk through the elements of opset_import field and register operator sets
            // for each domain. An exception UnknownDomain() will raise if the domain is
            // unknown or invalid.
            for (const auto& id : m_model_proto->opset_import())
            {
                m_opset.emplace(id.domain(),
                                OperatorsBridge::get_operator_set(
                                    id.version(), (id.domain() == "ai.onnx" ? "" : id.domain())));
            }
            // onnx.proto(.3): the empty string ("") for domain or absence of opset_import field
            // implies the operator set that is defined as part of the ONNX specification.
            const auto dm = m_opset.find("");
            if (dm == std::end(m_opset))
            {
                m_opset.emplace("", OperatorsBridge::get_operator_set(ONNX_OPSET_VERSION, ""));
            }
        }

        const Operator& Model::get_operator(const std::string& name,
                                            const std::string& domain) const
        {
            const auto dm = m_opset.find(domain);
            if (dm == std::end(m_opset))
            {
                throw error::UnknownDomain{domain};
            }
            const auto op = dm->second.find(name);
            if (op == std::end(dm->second))
            {
                throw error::UnknownOperator{name, domain};
            }
            return op->second;
        }

        bool Model::is_operator_available(const onnx::NodeProto& node_proto) const
        {
            const auto dm = m_opset.find(node_proto.domain());
            if (dm == std::end(m_opset))
            {
                return false;
            }
            const auto op = dm->second.find(node_proto.op_type());
            return (op != std::end(dm->second));
        }

    } // namespace onnx_import

} // namespace ngraph
