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

#include "ngraph/assertion.hpp"
#include "ngraph/except.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
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

        } // namespace  error

    } // namespace  onnx_import

} // namespace  ngraph

#define ASSERT_IS_SUPPORTED(node_, cond_)                                                          \
    NGRAPH_ASSERT_STREAM(ngraph::onnx_import::error::NotSupported, cond_) << (node_) << " "
#define ASSERT_VALID_ARGUMENT(node_, cond_)                                                        \
    NGRAPH_ASSERT_STREAM(ngraph::onnx_import::error::InvalidArgument, cond_) << (node_) << " "
