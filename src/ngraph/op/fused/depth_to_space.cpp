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
#include <cstddef>
#include <memory>

#include <iterator> //TODO

#include "depth_to_space.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::DepthToSpace::type_info;

op::DepthToSpace::DepthToSpace(const Output<Node>& data,
                               const DepthToSpaceMode& mode,
                               const size_t block_size)
    : FusedOp({data})
    , m_blocksize(block_size)
    , m_mode(mode)
{
    constructor_validate_and_infer_types();
}

op::DepthToSpace::DepthToSpace(const Output<Node>& data,
                               const std::string& mode,
                               const size_t block_size)
    : DepthToSpace(data, mode_from_string(mode), block_size)
{
}

static string vec_to_string(vector<size_t> in)
{
    std::ostringstream vts;

    if (!in.empty())
    {
        // Convert all but the last element to avoid a trailing ","
        std::copy(in.begin(), in.end() - 1, std::ostream_iterator<int>(vts, ", "));

        // Now add the last element with no delimiter
        vts << in.back();

        return vts.str();
    }
}

NodeVector op::DepthToSpace::decompose_op() const
{
    auto data = input_value(0);
    auto data_shape = data.get_shape();

    // TODO REMOVE h and w
    size_t n{1}, c{1}, h{1}, w{1};

    NGRAPH_CHECK((data_shape.size() >= 3),
                 "The input tensor with rank lower than 3 is not supported (input rank: ",
                 data_shape,
                 ")");

    if (data_shape.size() == 3)
    {
        // Insert batch axis
        data_shape.insert(data_shape.begin(), 1);
        data = builder::reshape(data, data_shape);
        c = data_shape.at(0);
        h = data_shape.at(1);
        w = data_shape.at(2);
    }

    n = data_shape.at(0);
    c = data_shape.at(1);
    h = data_shape.at(2);
    w = data_shape.at(3);

    NGRAPH_CHECK((c % (m_blocksize * m_blocksize) == 0 && m_blocksize > 0),
                 "SpaceToDepth: The depth axis size must be a multiple of ",
                 "squared block_size attribute value.");

    auto bs = static_cast<size_t>(m_blocksize);
    size_t c_flat = c / (bs * bs);
    const auto spatial_dims = data_shape.size() - 2;

    // First we have to disperse the data from depth channel, then rearrange them
    // so as appropriate chunks of data where close to their destination place.
    // Finally squeeze data from respective dimensions.
    shared_ptr<Node> flat_node;

    Shape dispersed_shape{n};
    for (int i = 0; i < data_shape.size(); ++i)
    {
        if (i < spatial_dims)
        {
            dispersed_shape.push_back(bs);
        }
        else
        {
            dispersed_shape.push_back(data_shape.at(i));
        }
    }
    vector<size_t> axes_order{0};
    switch (m_mode)
    {
    case DepthToSpaceMode::DEPTH_FIRST:
    {
        // reshape(data, [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2,
        // ..., DK])
        dispersed_shape.insert(dispersed_shape.begin() + 1, c_flat);
        std::cout << "1. DF: Expected: " << Shape{n, c_flat, bs, bs, h, w}
                  << " is: " << dispersed_shape << "\n";
        flat_node = builder::reshape(data, dispersed_shape);

        // transpose(x', [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1])
        axes_order.push_back(1);
        for (int i = 2; i < data_shape.size(); ++i)
        {
            axes_order.push_back(spatial_dims + i);
            axes_order.push_back(i);
        }

        std::cout << "2. DF: Expected: "
                  << "{0, 1, 4, 2, 5, 3}"
                  << " is: " << vec_to_string(axes_order) << "\n";
        flat_node = builder::reorder_axes(flat_node, axes_order);
        break;
    }
    case DepthToSpaceMode::BLOCKS_FIRST:
    default:
    {
        // reshape(data, [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2,
        // ..., DK])
        dispersed_shape.insert(dispersed_shape.begin() + spatial_dims, c_flat);
        std::cout << "1. BF: Expected: " << Shape{n, bs, bs, c_flat, h, w}
                  << " is: " << dispersed_shape << "\n";
        flat_node = builder::reshape(data, dispersed_shape);

        // transpose(x', [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K])
        axes_order.push_back(spatial_dims + 1);
        for (int i = 2; i < data_shape.size(); ++i)
        {
            axes_order.push_back(spatial_dims + i);
            axes_order.push_back(i - 1);
        }
        std::cout << "2. BF: Expected: "
                  << "{0, 3, 4, 1, 5, 2}"
                  << " is: " << vec_to_string(axes_order) << "\n";
        flat_node = builder::reorder_axes(flat_node, axes_order);
    }
    }
    // reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size, D3 * block_size,
    // ..., DK * block_size])
    Shape squeezed_shape{n, c_flat};
    for (int i = spatial_dims; i < data_shape.size(); ++i)
    {
        squeezed_shape.push_back(data_shape.at(i) * bs);
    }
    std::cout << "3. Expected: " << Shape{n, c_flat, h * bs, w * bs} << " is: " << squeezed_shape
              << "\n";
    return NodeVector{builder::reshape(flat_node, squeezed_shape)};
}

shared_ptr<Node> op::DepthToSpace::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<DepthToSpace>(new_args.at(0), m_mode, m_blocksize);
}

op::DepthToSpace::DepthToSpaceMode op::DepthToSpace::mode_from_string(const std::string& mode) const
{
    static const std::map<std::string, DepthToSpaceMode> allowed_values = {
        {"blocks_first", DepthToSpaceMode::BLOCKS_FIRST},
        {"depth_first", DepthToSpaceMode::DEPTH_FIRST}};

    NODE_VALIDATION_CHECK(
        this, allowed_values.count(mode) > 0, "Invalid 'depth_to_space_mode' value passed in.");

    return allowed_values.at(mode);
}
