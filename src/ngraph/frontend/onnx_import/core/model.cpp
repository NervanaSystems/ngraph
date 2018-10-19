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

#include <onnx-ml.pb.h>

#include "model.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        Model::Model(const onnx::ModelProto& model_proto)
            : m_model_proto{&model_proto}
        {
            for (const auto& id : m_model_proto->opset_import())
            {
                // onnx.proto(.3): the empty string ("") or absence of this field implies
                // the operator set that is defined as part of the ONNX specification.
                if (id.domain().empty())
                {
                    m_opset_version = id.version();
                }
            }
        }

    } // namespace onnx_import

} // namespace ngraph
