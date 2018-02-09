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

#pragma once

#include <memory>
#include <set>
#include <utility>
#include <vector>

// Names for types that aren't worth giving their own classes
namespace ngraph
{
    class Node;
    namespace op
    {
        class Parameter;

        /// A list of parameters
        using Parameters = std::vector<std::shared_ptr<Parameter>>;
    }
    class ValueType;

    /// @brief Zero or more value types
    using ValueTypes = std::vector<std::shared_ptr<const ValueType>>;

    /// @brief Zero or more nodes
    using Nodes = std::vector<std::shared_ptr<Node>>;

    /// @brief A sequence of axes
    using AxisVector = std::vector<size_t>;

    /// @brief A set of axes, for example, reduction axes
    using AxisSet = std::set<size_t>;

    /// @brief Coordinate in a tensor
    using Coordinate = std::vector<size_t>;

    /// @brief Shape for a tensor
    using Shape = std::vector<size_t>;

    /// @brief Strides of a tensor
    using Strides = std::vector<size_t>;

    /// @brief A coordinate-like type whose elements are allowed to be
    ///        negative.
    ///
    ///        Currently used only to express negative padding; in the future,
    ///        could conceivably be used to express
    using CoordinateDiff = std::vector<std::ptrdiff_t>;

    Coordinate project_coordinate(const Coordinate& coord, const AxisSet& deleted_axes);
    Shape project_shape(const Shape& shape, const AxisSet& deleted_axes);

    Coordinate inject_coordinate(const Coordinate& coord, size_t new_axis_pos, size_t new_axis_val);
    Coordinate inject_coordinate(const Coordinate& coord,
                                 std::vector<std::pair<size_t, size_t>> new_axis_pos_val_pairs);
    Shape inject_shape(const Shape& shape, size_t new_axis_pos, size_t new_axis_length);
    Shape inject_shape(const Shape& shape,
                       std::vector<std::pair<size_t, size_t>> new_axis_pos_length_pairs);
}
