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

#include <onnx.pb.h>
#include <ostream>

namespace ngraph
{
    namespace onnx_import
    {
        class Model
        {
        public:
            Model() = delete;
            explicit Model(const onnx::ModelProto& model_proto)
                : m_model_proto{&model_proto}
            {
            }

            Model(Model&&) noexcept = default;
            Model(const Model&) = default;

            Model& operator=(Model&&) noexcept = delete;
            Model& operator=(const Model&) = delete;

            const std::string& get_producer_name() const { return m_model_proto->producer_name(); }
            const onnx::GraphProto& get_graph() const { return m_model_proto->graph(); }
            std::int64_t get_model_version() const { return m_model_proto->model_version(); }
            const std::string& get_producer_version() const
            {
                return m_model_proto->producer_version();
            }

        private:
            const onnx::ModelProto* m_model_proto;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Model& model)
        {
            return (outs << "<Model: " << model.get_producer_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
