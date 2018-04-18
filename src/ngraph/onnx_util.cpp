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

#include <fstream>
#include <string>

#include "except.hpp"
#include "ngraph/onnx_util.hpp"

onnx::ModelProto ngraph::onnx_util::load_model_file(const std::string& filepath)
{
    onnx::ModelProto model_proto;

    {
        std::fstream input(filepath, std::ios::in | std::ios::binary);
        if (!input)
        {
            throw ngraph::ngraph_error("File not found: " + filepath);
        }
        else if (!model_proto.ParseFromIstream(&input))
        {
            throw ngraph::ngraph_error("Failed to parse ONNX file: " + filepath);
        }
    }

    return model_proto;
}
