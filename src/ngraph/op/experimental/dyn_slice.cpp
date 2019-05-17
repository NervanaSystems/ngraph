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

#include "ngraph/op/experimental/dyn_slice.hpp"

#include "ngraph/op/constant.hpp"

#include <memory>

using namespace std;
using namespace ngraph;

op::DynSlice::DynSlice(const shared_ptr<Node>& arg,
                       const shared_ptr<Node>& lower_bounds,
                       const shared_ptr<Node>& upper_bounds,
                       const shared_ptr<Node>& strides,
                       const AxisSet& lower_bounds_mask,
                       const AxisSet& upper_bounds_mask,
                       const AxisSet& new_axis,
                       const AxisSet& shrink_axis,
                       const AxisSet& ellipsis_mask)
    : Op("DynSlice", check_single_output_args({arg, lower_bounds, upper_bounds, strides}))
    , m_lower_bounds_mask(lower_bounds_mask)
    , m_upper_bounds_mask(upper_bounds_mask)
    , m_new_axis(new_axis)
    , m_shrink_axis(shrink_axis)
    , m_ellipsis_mask(ellipsis_mask)
{
    constructor_validate_and_infer_types();
}

Shape op::DynSlice::compute_output_shape() const
{
    auto input_shape = get_input_partial_shape(0).to_shape();
    auto lower_bounds = dynamic_pointer_cast<op::Constant>(get_argument(1));
    auto upper_bounds = dynamic_pointer_cast<op::Constant>(get_argument(2));
    auto strides = dynamic_pointer_cast<op::Constant>(get_argument(3));

    if (lower_bounds && upper_bounds && strides)
    {
        auto lb = lower_bounds->get_vector<int64_t>();
        auto ub = upper_bounds->get_vector<int64_t>();
        auto str = strides->get_vector<int64_t>();

        int max_dims = input_shape.size() + m_new_axis.size();
        if (lb.size() && ub.size())
        {
            NODE_VALIDATION_CHECK(
                this,
                lb.size() == ub.size(),
                "Lower bounds and Upper bounds needs to have same number of values");
        }
        if (lb.size() && str.size())
        {
            NODE_VALIDATION_CHECK(this,
                                  lb.size() == str.size(),
                                  "Lower bounds and strides needs to have same number of values");
        }
        if (ub.size() && str.size())
        {
            NODE_VALIDATION_CHECK(this,
                                  ub.size() == str.size(),
                                  "Upper bounds and strides needs to have same number of values");
        }

        int bounds_size =
            lb.size() ? lb.size() : (ub.size() ? ub.size() : (str.size() ? str.size() : 0));

        NODE_VALIDATION_CHECK(
            this, m_ellipsis_mask.size() <= 1, "Ellipsis mask cannot specify more than one axis");

        int ellipsis_pos1 = m_ellipsis_mask.size() ? *m_ellipsis_mask.begin() : max_dims;

        int ellipsis_pos2 = max_dims;
        bounds_size -= ellipsis_pos1;
        if (bounds_size > 0 && (max_dims - bounds_size) > ellipsis_pos1)
        {
            ellipsis_pos2 = max_dims - bounds_size;
        }

        std::vector<int> begin_dms(max_dims, 0);
        std::vector<int> end_dms(max_dims, -1);
        std::vector<int> stride_dms(max_dims, 1);

        int i, j, k, bj, ej, sj;
        Shape out_dims;
        for (i = 0, j = 0, k = 0, bj = 0, ej = 0, sj = 0; i < max_dims; i++)
        {
            if (i >= ellipsis_pos1 && i < ellipsis_pos2)
            {
                if (m_new_axis.find(i) == m_new_axis.end())
                {
                    end_dms[i] = end_dms[i] >= 0 ? end_dms[i] : input_shape[j++] + end_dms[i];
                }
                else
                {
                    end_dms[i] = begin_dms[i];
                }
                out_dims.push_back(
                    static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) /
                                          static_cast<float>(abs(stride_dms[i])))));
                k = ellipsis_pos1;
                continue;
            }
            stride_dms[i] = (str.size() > sj && str[sj] != 0) ? str[sj++] : 1;

            // Use lower_bounds if mask is not set
            if (m_lower_bounds_mask.find(j) == m_lower_bounds_mask.end())
            {
                begin_dms[i] = lb.size() > bj ? lb[bj] : (stride_dms[i] > 0 ? 0 : -1);
            }
            else
            {
                begin_dms[i] = stride_dms[i] > 0 ? 0 : -1;
            }
            bj++;

            begin_dms[i] = begin_dms[i] >= 0 ? begin_dms[i] : input_shape[j] + begin_dms[i];
            //  Clipping 'begin'
            begin_dms[i] =
                (begin_dms[i] < 0) ? 0 : (begin_dms[i] >= input_shape[j] ? input_shape[j] - 1
                                                                         : begin_dms[i]);

            // Use upper_bounds if mask is not set
            if (m_upper_bounds_mask.find(j) == m_upper_bounds_mask.end())
            {
                int end_dms_tmp =
                    ub.size() > ej ? (stride_dms[i] > 0 ? ub[ej] - 1 : ub[ej] + 1) : end_dms[i];
                end_dms[i] = ub.size() > ej ? end_dms_tmp : (stride_dms[i] > 0 ? -1 : 0);
            }
            else
            {
                end_dms[i] = stride_dms[i] > 0 ? -1 : 0;
            }
            ej++;
            end_dms[i] = end_dms[i] >= 0 ? end_dms[i] : input_shape[j] + end_dms[i];
            //  Clipping 'end'
            end_dms[i] = (end_dms[i] < 0) ? 0 : (end_dms[i] >= input_shape[j] ? input_shape[j] - 1
                                                                              : end_dms[i]);

            if (m_new_axis.find(i) == m_new_axis.end())
            {
                j++;
            }
            else
            {
                end_dms[i] = 0;
            }

            if (m_shrink_axis.find(k) != m_shrink_axis.end())
            {
                end_dms[i] = begin_dms[i];
            }
            else
            {
                out_dims.push_back(
                    static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) /
                                          static_cast<float>(abs(stride_dms[i])))));
            }

            k++;
        }
        return out_dims;
    }
    return Shape{};
}

void op::DynSlice::validate_and_infer_types()
{
    auto lower_bounds_et = get_input_element_type(1);
    auto upper_bounds_et = get_input_element_type(2);
    auto strides_et = get_input_element_type(3);

    // check data types
    NODE_VALIDATION_CHECK(this,
                          lower_bounds_et.compatible(element::Type_t::i64),
                          "Lower bounds must have element type i64.");
    NODE_VALIDATION_CHECK(this,
                          upper_bounds_et.compatible(element::Type_t::i64),
                          "Upper bounds must have element type i64.");
    NODE_VALIDATION_CHECK(
        this, strides_et.compatible(element::Type_t::i64), "Strides must have element type i64");

    // check shapes
    auto arg_shape = get_input_partial_shape(0);
    auto lower_bounds_shape = get_input_partial_shape(1);
    auto upper_bounds_shape = get_input_partial_shape(2);
    auto strides_shape = get_input_partial_shape(3);
    NODE_VALIDATION_CHECK(this,
                          lower_bounds_shape.rank().compatible(1),
                          "Lower bounds shape must have rank 1, got ",
                          lower_bounds_shape.rank(),
                          ".");
    NODE_VALIDATION_CHECK(this,
                          upper_bounds_shape.rank().compatible(1),
                          "Upper bounds shape must have rank 1, got ",
                          upper_bounds_shape.rank(),
                          ".");
    NODE_VALIDATION_CHECK(this,
                          strides_shape.rank().compatible(1),
                          "Strides shape must have rank 1, got ",
                          strides_shape.rank(),
                          ".");

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);

    if (get_input_partial_shape(0).is_static())
    {
        auto shape = compute_output_shape();
        if (shape != Shape{})
        {
            set_output_type(0, get_input_element_type(0), shape);
        }
        else
        {
            set_output_type(0, get_input_element_type(0), PartialShape::dynamic(arg_shape.rank()));
        }
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(arg_shape.rank()));
    }
}

shared_ptr<Node> op::DynSlice::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynSlice>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

void op::DynSlice::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("generate_adjoints not implemented for DynSlice");
}
