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

#include "ngraph/builder/autobroadcast.hpp"

#include <memory>
#include <numeric>
#include <sstream>

#include "ngraph/axis_vector.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"

using namespace std;

/// \brief Calculate the output shape of numpy-style broadcast operation for two shapes.
///
/// more info:
/// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
/// example:
/// left: [3, 1, 10] right: [5, 1]
/// return: [3, 5, 10]
///
/// \param left_shape First input shape.
/// \param right_shape Second input Shape.
/// \return Broadcast shape of input shapes.
static ngraph::Shape calculate_broadcast_shape(ngraph::Shape left_shape, ngraph::Shape right_shape)
{
    ngraph::Shape result;
    auto left_rank = left_shape.size();
    auto right_rank = right_shape.size();
    auto max_rank = std::max(left_rank, right_rank);

    // left-pad the left_shape with zeros
    left_shape.insert(std::begin(left_shape), max_rank - left_rank, 0);
    // left-pad the right_shape with zeros
    right_shape.insert(std::begin(right_shape), max_rank - right_rank, 0);

    for (std::size_t index = 0; index < max_rank; ++index)
    {
        result.push_back(std::max(left_shape.at(index), right_shape.at(index)));
    }

    return result;
};

/// \brief Calculate the output shape of numpy-style broadcast operation for all input shapes.
///
/// This function finds the maximum tensor shape that will be the result of element-wise operation
/// that will be applied to the input shapes vector. The function also prepares the shape of each
/// input for the element-wise operation by left-padding those shapes so that their rank is equal
/// to the left_shape's rank.
///
/// \param input_shapes A vector of input shapes for which a common shape should be found
/// \return A pair that contains the target shape as its first object and a vector of padded
///         input shapes ready to be broadcasted as the second object
static std::pair<ngraph::Shape, std::vector<ngraph::Shape>>
    get_numpy_broadcast_shapes(const std::vector<ngraph::Shape>& input_shapes)
{
    ngraph::Shape target_shape = std::accumulate(std::begin(input_shapes),
                                                 std::end(input_shapes),
                                                 ngraph::Shape{},
                                                 calculate_broadcast_shape);

    std::vector<ngraph::Shape> full_shapes;
    for (const ngraph::Shape& input : input_shapes)
    {
        ngraph::Shape padded_shape{input};
        padded_shape.insert(std::begin(padded_shape), target_shape.size() - padded_shape.size(), 1);
        full_shapes.push_back(std::move(padded_shape));
    }

    return {target_shape, full_shapes};
}

static std::pair<ngraph::Shape, std::vector<ngraph::Shape>>
    get_numpy_broadcast_shapes(const ngraph::OutputVector& values)
{
    std::vector<ngraph::Shape> input_shapes;

    for (const auto& input : values)
    {
        input_shapes.push_back(input.get_shape());
    }

    return get_numpy_broadcast_shapes(input_shapes);
}

/// \brief      Broadcast input node.
///
/// \note       The source shape does not have to be the actual shape of input node. However
///             it should be a superset of it (containing it as a continuous subset). This implies
///             we may expand the number of axes of input node.
///             The ranks of source_shape and output_shape must be equal. This means that the
///             source_shape has to be padded with ones for this operation.
///
/// \param[in]  value         The input Node to be broadcast.
/// \param[in]  output_shape  The output shape.
/// \param[in]  source_shape  The source shape from which we want to broadcast input node.
///
/// \return     The broadcasted Node.
///
static std::shared_ptr<ngraph::Node>
    broadcast_node_numpy_style(const ngraph::Output<ngraph::Node>& value,
                               const ngraph::Shape& output_shape,
                               const ngraph::Shape& source_shape)
{
    // If node already has the required shape, return original node
    if (output_shape == value.get_shape())
    {
        return value.as_single_output_node();
    }

    if (source_shape.size() != output_shape.size())
    {
        NGRAPH_WARN << "Ranks of source_shape and output_shape dont match: " << source_shape.size()
                    << " vs " << output_shape.size();
    }

    ngraph::AxisVector broadcast_axes;
    ngraph::Shape squeezed_shape;
    // Positions of axes which have length of 1 are needed to calculate broadcast_axes
    // for nGraph broadcast operation. We need to remove all ones from source shape
    // to avoid broadcasting axis conflict.
    for (std::size_t index = 0; index < output_shape.size(); ++index)
    {
        if (source_shape.at(index) == 1)
        {
            broadcast_axes.push_back(index);
        }
        else
        {
            squeezed_shape.push_back(source_shape.at(index));
        }
    }

    // Remove axes which have length of 1 from source shape
    ngraph::Output<ngraph::Node> broadcasted_value = std::make_shared<ngraph::op::Reshape>(
        value, ngraph::get_default_order(value.get_shape()), squeezed_shape);

    return std::make_shared<ngraph::op::Broadcast>(broadcasted_value, output_shape, broadcast_axes);
}

/// \brief      Broadcast input node.
///
/// \param[in]  value         The input Node to be broadcast.
/// \param[in]  output_shape  The output shape.
/// \param[in]  axis          The start index to align with output_shape
///
/// \return     The broadcasted Node.
///
static std::shared_ptr<ngraph::Node> broadcast_value_pdpd_style(
    const ngraph::Output<ngraph::Node>& value, const ngraph::Shape& output_shape, int64_t axis)
{
    auto value_shape = value.get_shape();

    // If node already has the required shape, return original node
    if (output_shape == value_shape)
    {
        return value.as_single_output_node();
    }

    if (axis == -1)
    {
        axis = output_shape.size() - value_shape.size();
    }

    auto trimmed_value_shape = value_shape;
    while (trimmed_value_shape.size() > 0 && trimmed_value_shape.back() == 1)
    {
        trimmed_value_shape.pop_back();
    }

    ngraph::AxisSet axes;
    for (int64_t i = 0; i < axis; ++i)
    {
        axes.insert(static_cast<size_t>(i));
    }

    for (size_t i = axis + trimmed_value_shape.size(); i < output_shape.size(); ++i)
    {
        axes.insert(i);
    }

    auto trimmed_value = value;
    if (value_shape != trimmed_value_shape)
    {
        trimmed_value = std::make_shared<ngraph::op::Reshape>(
            value, ngraph::get_default_order(value_shape), trimmed_value_shape);
    }

    auto value_bcast = std::make_shared<ngraph::op::Broadcast>(trimmed_value, output_shape, axes);

    return std::move(value_bcast);
}

namespace ngraph
{
    namespace builder
    {
        numpy_autobroadcast_incompatible_shapes::numpy_autobroadcast_incompatible_shapes(
            const ngraph::Shape& shape1, const ngraph::Shape& shape2)
            : ngraph::ngraph_error(error_str(shape1, shape2))
            , m_shape1(shape1)
            , m_shape2(shape2)
        {
        }

        std::string numpy_autobroadcast_incompatible_shapes::error_str(const ngraph::Shape& shape1,
                                                                       const ngraph::Shape& shape2)
        {
            ostringstream os;
            os << "Auto-broadcast not possible for these input shapes:"
               << " shape1=" << vector_to_string(shape1) << " shape2=" << vector_to_string(shape2);
            return os.str();
        }

        /// A utility struct representing the details computed by the
        /// compute_shapes_and_broadcast_axes function.
        struct Autobroadcast_plan
        {
            ngraph::Shape m_arg1_shape_after_possible_reshaping;
            ngraph::Shape m_arg2_shape_after_possible_reshaping;
            ngraph::AxisSet m_arg1_broadcast_axes;
            ngraph::AxisSet m_arg2_broadcast_axes;
            ngraph::Shape m_final_shape;
        };

        /// \brief Compute the details regarding what reshape and/or broadcast operations must be
        ///        applied to arg1 and/or arg2, as well as what the final resulting shape shall
        ///        be.
        ///
        /// If this algorithm cannot handle the particular combination of shapes supplied as
        /// inputs, throw an ngraph::builder::numpy_autobroadcast_incompatible_shapes exception.
        ///
        /// \exception ngraph::builder::numpy_autobroadcast_incompatible_shapes
        static Autobroadcast_plan
            compute_shapes_and_broadcast_axes(const ngraph::Shape& arg1_in_shape,
                                              const ngraph::Shape& arg2_in_shape)
        {
            Autobroadcast_plan plan;

            size_t arg1_size = arg1_in_shape.size();
            size_t arg2_size = arg2_in_shape.size();
            size_t axis = std::max(arg1_size, arg2_size) - 1;

            // per numpy definition of broadcast:
            // start with trailing dimensions and work forward
            // two dimensions are compatible:
            //  * if they are equal
            //  * if one of them is 1
            while (arg1_size >= 1 || arg2_size >= 1)
            {
                size_t arg1_dim = arg1_size ? arg1_in_shape[arg1_size - 1] : 1;
                size_t arg2_dim = arg2_size ? arg2_in_shape[arg2_size - 1] : 1;

                if (arg1_dim == arg2_dim)
                {
                    // add dimension to broadcast shape + arg1/arg2 reshape
                    plan.m_final_shape.insert(plan.m_final_shape.begin(), arg1_dim);
                    plan.m_arg1_shape_after_possible_reshaping.insert(
                        plan.m_arg1_shape_after_possible_reshaping.begin(), arg1_dim);
                    plan.m_arg2_shape_after_possible_reshaping.insert(
                        plan.m_arg2_shape_after_possible_reshaping.begin(), arg2_dim);
                }
                else if (arg2_dim == 1)
                {
                    // add arg1 dimension to broadcast shape and arg1 reshape
                    plan.m_final_shape.insert(plan.m_final_shape.begin(), arg1_dim);
                    plan.m_arg1_shape_after_possible_reshaping.insert(
                        plan.m_arg1_shape_after_possible_reshaping.begin(), arg1_dim);
                    // add current axis to arg2 broadcast axes
                    plan.m_arg2_broadcast_axes.insert(plan.m_arg2_broadcast_axes.begin(), axis);
                }
                else if (arg1_dim == 1)
                {
                    // add arg2 dimension to broadcast shape and arg2 reshape
                    plan.m_final_shape.insert(plan.m_final_shape.begin(), arg2_dim);
                    plan.m_arg2_shape_after_possible_reshaping.insert(
                        plan.m_arg2_shape_after_possible_reshaping.begin(), arg2_dim);
                    // add current axis to arg1 broadcast axes
                    plan.m_arg1_broadcast_axes.insert(plan.m_arg1_broadcast_axes.begin(), axis);
                }
                else
                {
                    throw numpy_autobroadcast_incompatible_shapes(arg1_in_shape, arg2_in_shape);
                }

                if (arg1_size)
                {
                    --arg1_size;
                }

                if (arg2_size)
                {
                    --arg2_size;
                }

                if (axis)
                {
                    --axis;
                }
            }

            return plan;
        }

        /// If necessary, wrap \p node with an additional reshape and/or broadcast op.
        /// Return a pointer to the node that produces the wrapped value.
        /// If no additional reshape or broadcast op was needed, simply return \p node.
        static std::shared_ptr<Node>
            add_required_ops(const Output<Node>& value,
                             const ngraph::Shape& shape_after_possible_reshaping,
                             const ngraph::AxisSet& broadcast_axes,
                             const ngraph::Shape& final_shape)
        {
            Output<Node> return_value{value};

            if (value.get_shape() != shape_after_possible_reshaping)
            {
                // tell reshape to examine input dimensions in order
                ngraph::AxisVector order = ngraph::get_default_order(value.get_shape());
                return_value = std::make_shared<ngraph::op::Reshape>(
                    return_value, order, shape_after_possible_reshaping);
            }

            if (final_shape != shape_after_possible_reshaping)
            {
                return_value = std::make_shared<ngraph::op::Broadcast>(
                    return_value, final_shape, broadcast_axes);
            }

            return return_value.get_node_shared_ptr()->add_provenance_group_members_above({value});
        }

        std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>
            numpy_broadcast(const std::pair<Output<Node>, Output<Node>>& args)
        {
            NGRAPH_CHECK(args.first.get_node());
            NGRAPH_CHECK(args.second.get_node());

            const ngraph::Shape& arg1_in_shape = args.first.get_shape();
            const ngraph::Shape& arg2_in_shape = args.second.get_shape();

            // Handle the trivial case...
            if (arg1_in_shape == arg2_in_shape)
            {
                return make_pair(args.first.as_single_output_node(),
                                 args.second.as_single_output_node());
            }

            Autobroadcast_plan plan =
                compute_shapes_and_broadcast_axes(arg1_in_shape, arg2_in_shape);

            auto arg1_out = add_required_ops(args.first,
                                             plan.m_arg1_shape_after_possible_reshaping,
                                             plan.m_arg1_broadcast_axes,
                                             plan.m_final_shape);

            auto arg2_out = add_required_ops(args.second,
                                             plan.m_arg2_shape_after_possible_reshaping,
                                             plan.m_arg2_broadcast_axes,
                                             plan.m_final_shape);

            return {arg1_out, arg2_out};
        }

        //-----------------------------------------------------------------------------------------
        //
        //                      ONNX HELPER FUNCTIONS
        //
        //-----------------------------------------------------------------------------------------

        OutputVector numpy_broadcast_outputs(const OutputVector& values)
        {
            if (values.size() <= 1)
            {
                return values;
            }

            // find the output tensor's shape, then broadcast all inputs so that they are compatible
            auto bcast_shapes = get_numpy_broadcast_shapes(values);

            OutputVector broadcasted_inputs;
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                broadcasted_inputs.push_back(broadcast_node_numpy_style(
                    values[i], bcast_shapes.first, bcast_shapes.second[i]));
            }
            return broadcasted_inputs;
        }

        std::shared_ptr<ngraph::Node> numpy_broadcast(const Output<ngraph::Node>& value,
                                                      const Shape& shape)
        {
            auto bcast_shape = get_numpy_broadcast_shapes({value.get_shape(), shape});
            return broadcast_node_numpy_style(value, bcast_shape.first, bcast_shape.second[0]);
        }

        OutputVector numpy_broadcast_for_matmul_operation(const Output<ngraph::Node>& left,
                                                          const Output<ngraph::Node>& right)
        {
            const auto& left_shape = left.get_shape();
            const auto& right_shape = right.get_shape();
            // Broadcast only _stack of matrices_ axes.
            const auto& numpy_shapes = get_numpy_broadcast_shapes(
                {Shape{std::begin(left_shape), std::next(std::end(left_shape), -2)},
                 Shape{std::begin(right_shape), std::next(std::end(right_shape), -2)}});

            // Prepare tensors output shapes with broadcasted _stack of matrices_ axes.
            auto left_output_shape = numpy_shapes.first;
            auto right_output_shape = numpy_shapes.first;
            // Append the last two axes original dimensions.
            left_output_shape.insert(std::end(left_output_shape),
                                     std::next(std::begin(left_shape), left_shape.size() - 2),
                                     std::end(left_shape));
            right_output_shape.insert(std::end(right_output_shape),
                                      std::next(std::begin(right_shape), right_shape.size() - 2),
                                      std::end(right_shape));

            auto left_full_shape = numpy_shapes.second.at(0);
            auto right_full_shape = numpy_shapes.second.at(1);
            // Append the last two axes original dimensions.
            left_full_shape.insert(std::end(left_full_shape),
                                   std::next(std::begin(left_shape), left_shape.size() - 2),
                                   std::end(left_shape));
            right_full_shape.insert(std::end(right_full_shape),
                                    std::next(std::begin(right_shape), right_shape.size() - 2),
                                    std::end(right_shape));

            return {broadcast_node_numpy_style(left, left_output_shape, left_full_shape),
                    broadcast_node_numpy_style(right, right_output_shape, right_full_shape)};
        }

        OutputVector legacy_broadcast_for_binary_operation(const Output<ngraph::Node>& left,
                                                           const Output<ngraph::Node>& right,
                                                           size_t start_match_axis)
        {
            const auto& left_shape = left.get_shape();
            const auto& right_shape = right.get_shape();

            bool dimensions_identical = (left_shape == right_shape);
            if (dimensions_identical)
            {
                return {left, right};
            }

            // Prepare new shape of right operand for broadcasting
            // Remove dimensions with length=1 from back
            auto new_right_shape = right_shape;
            for (int dimension = new_right_shape.size() - 1; dimension >= 0; --dimension)
            {
                if (new_right_shape[dimension] == 1)
                {
                    new_right_shape.pop_back();
                }
                else
                {
                    break;
                }
            }

            // Find first dimensions at front with length different from 1
            std::size_t num_ones = 0;
            for (std::size_t dimension : new_right_shape)
            {
                if (dimension == 1)
                {
                    ++num_ones;
                }
                else
                {
                    break;
                }
            }

            // Remove dimensions with length=1 from front
            new_right_shape.erase(std::begin(new_right_shape),
                                  std::next(std::begin(new_right_shape), num_ones));

            auto reshape_right = std::make_shared<ngraph::op::Reshape>(
                right, ngraph::get_default_order(right_shape), new_right_shape);

            // Move broadcast start axis parameter to right
            start_match_axis += num_ones;

            auto broadcast_right = std::make_shared<ngraph::op::Broadcast>(
                reshape_right,
                left_shape,
                calculate_broadcast_axes(left_shape, new_right_shape, start_match_axis));

            return {left, broadcast_right};
        }

        OutputVector pdpd_broadcast(const OutputVector& inputs, int64_t axis)
        {
            if (inputs.size() <= 1)
            {
                return inputs;
            }

            OutputVector broadcasted_inputs{inputs[0]};
            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                broadcasted_inputs.push_back(
                    broadcast_value_pdpd_style(inputs[i], inputs[0].get_shape(), axis));
            }
            return broadcasted_inputs;
        }

        AxisSet calculate_broadcast_axes(const Shape& output_shape,
                                         const Shape& input_shape,
                                         std::size_t start_match_axis)
        {
            std::vector<std::size_t> result(output_shape.size() - input_shape.size());
            // Populate the result vector with monotonic increasing series from 0 until
            // output_shape_size, excluding values in range:
            // [start_match_axis, start_match_axis + input_shape.size()]
            std::iota(std::begin(result), std::begin(result) + start_match_axis, 0);
            std::iota(std::begin(result) + start_match_axis,
                      std::end(result),
                      start_match_axis + input_shape.size());
            return result;
        }

        namespace opset1
        {
            Output<Node> legacy_style_broadcast_for_binary_operation(const Output<Node>& left,
                                                                     const Output<Node>& right,
                                                                     size_t start_match_axis)
            {
                const auto& left_shape = left.get_shape();
                const auto& right_shape = right.get_shape();

                bool dimensions_identical = (left_shape == right_shape);
                if (dimensions_identical)
                {
                    return right;
                }

                // Prepare new shape of right operand for broadcasting
                // Remove dimensions with length=1 from back
                auto new_right_shape = right_shape;
                for (int dimension = new_right_shape.size() - 1; dimension >= 0; --dimension)
                {
                    if (new_right_shape.at(dimension) == 1)
                    {
                        new_right_shape.pop_back();
                    }
                    else
                    {
                        break;
                    }
                }

                // Find first dimensions at front with length different from 1
                std::size_t num_ones = 0;
                for (std::size_t dimension : new_right_shape)
                {
                    if (dimension == 1)
                    {
                        ++num_ones;
                    }
                    else
                    {
                        break;
                    }
                }

                // Remove dimensions with length=1 from front
                new_right_shape.erase(std::begin(new_right_shape),
                                      std::next(std::begin(new_right_shape), num_ones));

                auto reshape_right = reshape(right, new_right_shape);

                // Move broadcast start axis parameter to right
                start_match_axis += num_ones;

                return make_broadcast(reshape_right, left_shape, start_match_axis);
            }

            std::vector<std::size_t> get_axes_mapping(const Shape& output_shape,
                                                      const AxisSet& broadcast_axes)
            {
                NGRAPH_CHECK((broadcast_axes.size() <= output_shape.size()));
                std::vector<size_t> axes_mapping(output_shape.size());
                std::iota(axes_mapping.begin(), axes_mapping.end(), 0);
                for (auto i = broadcast_axes.rbegin(); i != broadcast_axes.rend(); ++i)
                {
                    axes_mapping.erase(axes_mapping.begin() + *i);
                }
                return axes_mapping;
            }

            Output<Node> get_axes_mapping_output(const Shape& output_shape,
                                                 const Shape& input_shape,
                                                 std::size_t start_match_axis)
            {
                NGRAPH_CHECK((input_shape.size() + start_match_axis <= output_shape.size()));
                std::vector<std::size_t> mapping(input_shape.size());
                std::iota(std::begin(mapping), std::end(mapping), start_match_axis);

                return ngraph::op::Constant::create(element::i64, Shape{mapping.size()}, mapping);
            }

            Output<Node> get_axes_mapping_output(const Shape& output_shape,
                                                 const AxisSet& broadcast_axes)
            {
                std::vector<size_t> axes_mapping{get_axes_mapping(output_shape, broadcast_axes)};
                return ngraph::op::Constant::create(
                    element::i64, Shape{axes_mapping.size()}, axes_mapping);
            }

            Output<Node> make_broadcast(const Output<Node>& node,
                                        const Shape& target_shape,
                                        const AxisSet& broadcast_axes)
            {
                return std::make_shared<ngraph::op::v1::Broadcast>(
                    node,
                    ngraph::op::Constant::create(
                        element::i64, Shape{target_shape.size()}, target_shape),
                    get_axes_mapping_output(target_shape, broadcast_axes));
            }

            Output<Node> make_broadcast(const Output<Node>& node,
                                        const Shape& target_shape,
                                        std::size_t start_match_axis)
            {
                return std::make_shared<ngraph::op::v1::Broadcast>(
                    node,
                    ngraph::op::Constant::create(
                        element::i64, Shape{target_shape.size()}, target_shape),
                    get_axes_mapping_output(target_shape, node.get_shape(), start_match_axis));
            }

        } // namespace opset1
    }     // namespace builder
} // namespace ngraph
