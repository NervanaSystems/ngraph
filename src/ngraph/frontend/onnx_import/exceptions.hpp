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

#pragma once

#include <string>

#include "core/node.hpp"
#include "ngraph/assertion.hpp"
#include "ngraph/check.hpp"
#include "ngraph/except.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            namespace detail
            {
                std::string get_error_msg_prefix(const Node& node);
            }

            struct NotSupported : AssertionFailure
            {
                explicit NotSupported(const std::string& what_arg)
                    : AssertionFailure(what_arg)
                {
                }
            };

            struct InvalidArgument : AssertionFailure
            {
                explicit InvalidArgument(const std::string& what_arg)
                    : AssertionFailure(what_arg)
                {
                }
            };

            class NodeValidationFailure : public CheckFailure
            {
            public:
                NodeValidationFailure(const CheckLocInfo& check_loc_info,
                                      const Node& node,
                                      const std::string& explanation)
                    : CheckFailure(check_loc_info, detail::get_error_msg_prefix(node), explanation)
                {
                }
            };

        } // namespace  error

    } // namespace  onnx_import

} // namespace  ngraph

#define ASSERT_IS_SUPPORTED(node_, cond_)                                                          \
    NGRAPH_ASSERT_STREAM_DO_NOT_USE_IN_NEW_CODE(ngraph::onnx_import::error::NotSupported, cond_)   \
        << (node_) << " "
#define ASSERT_VALID_ARGUMENT(node_, cond_)                                                        \
    NGRAPH_ASSERT_STREAM_DO_NOT_USE_IN_NEW_CODE(ngraph::onnx_import::error::InvalidArgument,       \
                                                cond_)                                             \
        << (node_) << " "

#define CHECK_VALID_NODE(node_, cond_, ...)                                                        \
    NGRAPH_CHECK_HELPER(                                                                           \
        ::ngraph::onnx_import::error::NodeValidationFailure, (node_), (cond_), ##__VA_ARGS__)
