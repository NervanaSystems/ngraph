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

#include <onnx-ml.pb.h>

#include <ostream>
#include <string>
#include <unordered_map>

#include "operator_set.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class Model
        {
        public:
            Model() = delete;
            explicit Model(const onnx::ModelProto& model_proto);

            Model(const Model&) = default;
            Model(Model&&) = default;

            Model& operator=(const Model&) = delete;
            Model& operator=(Model&&) = delete;

            const std::string& get_producer_name() const { return m_model_proto->producer_name(); }
            const onnx::GraphProto& get_graph() const { return m_model_proto->graph(); }
            std::int64_t get_model_version() const { return m_model_proto->model_version(); }
            const std::string& get_producer_version() const
            {
                return m_model_proto->producer_version();
            }

            /// \brief Access an operator object by its type name and domain name
            /// The function will return the operator object if it exists, or report an error
            /// in case of domain or operator absence.
            /// \param name       type name of the operator object,
            /// \param domain     domain name of the operator object.
            /// \return Reference to the operator object.
            /// \throw error::UnknownDomain    there is no operator set defined for the given domain,
            /// \throw error::UnknownOperator  the given operator type name does not exist in operator set.
            const Operator& get_operator(const std::string& name, const std::string& domain) const;

            /// \brief Check availability of operator base on NodeProto.
            /// \return `true` if the operator is available, otherwise it returns `false`.
            bool is_operator_available(const onnx::NodeProto& node_proto) const;

        private:
            const onnx::ModelProto* m_model_proto;
            std::unordered_map<std::string, OperatorSet> m_opset;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Model& model)
        {
            return (outs << "<Model: " << model.get_producer_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
