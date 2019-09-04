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

#include "core/node.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                /// \brief      Permutes input tensor data from depth into blocks of spatial data.
                ///
                /// \note       Values from the depth dimension (assuming NCHW layout) are moved in
                ///             spatial blocks to the height and width dimensions.
                ///
                /// \param[in]  node  The ONNX input node describing operation.
                ///
                /// \return     NodeVector containing Tensor with shape:
                ///             [N, C/(blocksize * blocksize), H * blocksize, W * blocksize]
                NodeVector depth_to_space(const Node& node);
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
