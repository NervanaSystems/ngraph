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

#include <algorithm>

#include "ngraph/common.hpp"

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

// TODO: check validity, i.e. that the new axis indices are all < coord_size+num_new_axes.
Coordinate ngraph::inject_coordinate(const Coordinate& coord,
                                     std::vector<std::pair<size_t, size_t>> new_axis_pos_val_pairs)
{
    Coordinate result;

    size_t original_pos = 0;

    for (size_t result_pos = 0; result_pos < coord.size() + new_axis_pos_val_pairs.size();
         result_pos++)
    {
        auto search_it = std::find_if(
            new_axis_pos_val_pairs.begin(),
            new_axis_pos_val_pairs.end(),
            [result_pos](std::pair<size_t, size_t> p) { return p.first == result_pos; });

        if (search_it == new_axis_pos_val_pairs.end())
        {
            result.push_back(coord[original_pos++]);
        }
        else
        {
            result.push_back(search_it->second);
        }
    }

    return result;
}

Coordinate
    ngraph::inject_coordinate(const Coordinate& coord, size_t new_axis_pos, size_t new_axis_val)
{
    return inject_coordinate(coord,
                             std::vector<std::pair<size_t, size_t>>{
                                 std::pair<size_t, size_t>(new_axis_pos, new_axis_val)});
}

Shape ngraph::inject_shape(const Shape& shape, size_t new_axis_pos, size_t new_axis_length)
{
    return inject_coordinate(shape, new_axis_pos, new_axis_length);
}

Shape inject_shape(const Shape& shape,
                   std::vector<std::pair<size_t, size_t>> new_axis_pos_length_pairs)
{
    return inject_coordinate(shape, new_axis_pos_length_pairs);
}
