// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/common.hpp"
#include "ngraph/view.hpp"

using namespace ngraph;

// TODO: check validity, i.e. that all deleted_axes are valid
Coordinate ngraph::project_coordinate(const Coordinate& coord, const AxisSet& deleted_axes)
{
    Coordinate result;

    for (size_t i = 0; i < coord.size(); i++)
    {
        if (deleted_axes.find(i) == deleted_axes.end())
        {
            result.push_back(coord[i]);
        }
    }

    return result;
}

Shape ngraph::project_shape(const Shape& shape, const AxisSet& deleted_axes)
{
    return project_coordinate(shape, deleted_axes);
}

// TODO: for the moment, just one axis at a time, please. Later could pass in an std::map from axis positions to axis lengths.
// TODO: check validity, i.e. that the new axis is < coord_size+1.
Coordinate
    ngraph::inject_coordinate(const Coordinate& coord, size_t new_axis_pos, size_t new_axis_val)
{
    Coordinate result;

    size_t original_pos = 0;

    for (size_t result_pos = 0; result_pos < coord.size() + 1; result_pos++)
    {
        if (result_pos == new_axis_pos)
        {
            result.push_back(new_axis_val);
        }
        else
        {
            result.push_back(coord[original_pos++]);
        }
    }

    return result;
}

Shape ngraph::inject_shape(const Shape& shape, size_t new_axis_pos, size_t new_axis_length)
{
    return inject_coordinate(shape, new_axis_pos, new_axis_length);
}
