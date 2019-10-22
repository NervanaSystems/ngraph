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

#include <algorithm>

#include "ngraph/shape_util.hpp"

using namespace ngraph;

template <>
PartialShape ngraph::project(const PartialShape& shape, const AxisSet& axes)
{
    if (shape.rank().is_dynamic())
    {
        return shape;
    }
    else
    {
        std::vector<Dimension> result_dims;

        for (size_t const& axe : axes)
        {
            if (axe >= size_t(shape.rank()))
            {
                break;
            }
            result_dims.push_back(shape[axe]);
        }

        return PartialShape(result_dims);
    }
}

template <>
PartialShape ngraph::reduce(const PartialShape& shape, const AxisSet& deleted_axes)
{
    if (shape.rank().is_dynamic())
    {
        return shape;
    }
    else
    {
        AxisSet axes;

        if (deleted_axes.empty())
        {
            for (size_t i = 0; i < size_t(shape.rank()); i++)
            {
                axes.insert(i);
            }
        }
        else
        {
            auto deleted_axes_it = deleted_axes.cbegin();
            for (size_t i = 0; i < size_t(shape.rank()); ++i)
            {
                if (*deleted_axes_it == i)
                {
                    if (++deleted_axes_it == deleted_axes.cend())
                    {
                        for (i = i + 1; i < size_t(shape.rank()); ++i)
                        {
                            axes.insert(i);
                        }
                        break;
                    }
                }
                else
                {
                    axes.insert(i);
                }
            }
        }

        return project(shape, axes);
    }
}

template <>
PartialShape
    ngraph::inject_pairs(const PartialShape& shape,
                         std::vector<std::pair<size_t, Dimension>> new_axis_pos_value_pairs)
{
    if (shape.rank().is_dynamic())
    {
        return shape;
    }
    else
    {
        std::vector<Dimension> result_dims;

        size_t original_pos = 0;

        for (size_t result_pos = 0;
             result_pos < size_t(shape.rank()) + new_axis_pos_value_pairs.size();
             result_pos++)
        {
            auto search_it = std::find_if(
                new_axis_pos_value_pairs.begin(),
                new_axis_pos_value_pairs.end(),
                [result_pos](std::pair<size_t, Dimension> p) { return p.first == result_pos; });

            if (search_it == new_axis_pos_value_pairs.end())
            {
                result_dims.push_back(shape[original_pos++]);
            }
            else
            {
                result_dims.push_back(search_it->second);
            }
        }

        return PartialShape{result_dims};
    }
}
