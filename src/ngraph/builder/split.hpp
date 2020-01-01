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

#pragma once
#include <memory>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        /// \brief      Split value on specified axis into multiple parts.
        ///
        /// \param[in]  value         The value to be split.
        /// \param[in]  length_parts  The vector defining the lengths of each split part.
        /// \param[in]  axis          The axis we split input node on. Default value is zero axis.
        ///
        /// \return     The vector containing multiple nodes we split input node into.
        ///
        NodeVector split(const Output<Node>& value,
                         const std::vector<size_t>& length_parts,
                         size_t axis = 0);

        /// \brief      Split node on specified axis into multiple parts.
        ///
        /// \param[in]  value         The value to split.
        /// \param[in]  split_parts   The number of parts we want to split output at given
        ///                           axis. The length of the axis to split must be divisible by
        ///                           this value.
        /// \param[in]  axis          The axis we split input node on. Default value is zero axis.
        ///
        /// \note       This implementation supports negative `axis` values (similar to NumPy
        ///             indexing). This means that the axis to split on will be counted from
        ///             the back of the tensor (negative values are subtracted from its rank).
        ///
        /// \return     The vector containing multiple nodes we split input node into.
        ///
        NodeVector split(const Output<Node>& value, size_t split_parts, int axis = 0);
    } // namespace builder
} // namespace ngraph
