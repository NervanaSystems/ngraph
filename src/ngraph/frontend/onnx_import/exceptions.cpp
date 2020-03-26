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

#include <sstream>

#include "exceptions.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            namespace detail
            {
                std::string get_error_msg_prefix(const Node& node)
                {
                    std::stringstream ss;
                    ss << "While validating ONNX node '" << node << "'";
                    return ss.str();
                }
            }
        }

        namespace validation
        {
            void check_valid_inputs_size(const onnx_import::Node& node, size_t minimum_inputs_size)
            {
                const auto inputs_size = node.get_ng_inputs().size();
                CHECK_VALID_NODE(node,
                                 inputs_size >= minimum_inputs_size,
                                 " Minimum required inputs size is: ",
                                 minimum_inputs_size,
                                 " Got: ",
                                 inputs_size);
            }
        }
    }
}
