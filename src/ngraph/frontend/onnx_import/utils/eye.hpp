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

#include <memory>

#include "common.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace eye
        {
            /// \brief Creates a square identity matrix.
            ///
            /// \param[in] n Order of the resulting matrix.
            ///
            /// \return A Constant node representing identity matrix with shape (n, n).
            template <typename T = double>
            std::shared_ptr<ngraph::op::Constant> square_identity(const size_t n,
                                                                  const element::Type& type)
            {
                std::vector<T> identity_matrix(n * n, T{0});

                for (size_t row = 0; row < n; ++row)
                {
                    const size_t diagonal_element = (n * row) + row;
                    identity_matrix.at(diagonal_element) = T{1};
                }

                return std::make_shared<ngraph::op::Constant>(type, Shape{{n, n}}, identity_matrix);
            }

        } //namespace eye

    } // namespace onnx_import

} // namespace ngraph
