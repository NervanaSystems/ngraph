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
#include <cmath>

#include <iterator> //TODO

#include "ngraph/builder/reshape.hpp"
#include "ngraph/shape.hpp"
#include "space_to_depth.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::SpaceToDepth::type_info;

op::SpaceToDepth::SpaceToDepth(const Output<Node>& data,
                               const SpaceToDepthMode& mode,
                               size_t block_size)
    : FusedOp({data})
    , m_blocksize(block_size)
    , m_mode(mode)
{
    constructor_validate_and_infer_types();
}

op::SpaceToDepth::SpaceToDepth(const Output<Node>& data, const std::string& mode, size_t block_size)
    : SpaceToDepth(data, mode_from_string(mode), block_size)
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

NodeVector op::SpaceToDepth::decompose_op() const
{
    auto data = input_value(0);
    auto data_shape = data.get_shape();

    NGRAPH_CHECK((data_shape.size() >= 3),
        "The input tensor with rank lower than 3 is not supported (input rank: ",
        data_shape.size(),
        ")");

    NGRAPH_CHECK(m_blocksize > 0, "m_blocksize must be greater than 0");

    if (data_shape.size() == 3)
    {
        // Insert batch axis
        data_shape.insert(data_shape.begin(), 1);
        data = builder::reshape(data, data_shape);
    }

    const size_t n_dim = data_shape.at(0);
    const size_t c_dim = data_shape.at(1);
    const size_t no_spatial_dims = 2;
    const size_t spatial_dims = data_shape.size() - no_spatial_dims;

    //TODO REMOVE
    size_t bs = static_cast<size_t>(m_blocksize);
    size_t n = data_shape.at(0);
    size_t c = data_shape.at(1);
    size_t h = data_shape.at(2);
    size_t w = data_shape.at(3);
    size_t w_flat = w / bs;
    size_t h_flat = h / bs;
    size_t c_high = c * bs * bs;

    for (int i = no_spatial_dims; i < data_shape.size(); ++i)
    {
        NGRAPH_CHECK(m_blocksize > 0 && data_shape.at(i) % m_blocksize == 0,
            "The dimension on position: ", i, " equal to: ", data_shape.at(i),
            " must be a multiple of m_blocksize: ", m_blocksize);
    }

    //TODO

    // First we have to disperse the data from height and width channels, then
    // rearrange them so as appropriate chunks of data where close to their
    // destination place. Finally squeeze data from respective dimensions.
    Shape dispersed_shape{ n_dim, c_dim };
    for (int i = 0; i < spatial_dims; ++i) {
        dispersed_shape.push_back(data_shape.at(i+no_spatial_dims) / m_blocksize);
        dispersed_shape.push_back(m_blocksize);
    }
    std::cout << "1. Expected: " << Shape{ n, c, h_flat, bs, w_flat, bs }
    << " is: " << dispersed_shape << "\n";
    auto flat_node = builder::reshape(data, dispersed_shape);
    vector<size_t> axes_order{ 0 };
    for (size_t i = 0, j = 3; i < spatial_dims; ++i, j += 2) {
        axes_order.push_back(j);
    }
    for (size_t i = 0, j = 2; i < spatial_dims; ++i, j += 2) {
        axes_order.push_back(j);
    }

    switch (m_mode)
    {
    case SpaceToDepthMode::DEPTH_FIRST:
    {
        axes_order.insert(axes_order.begin() + 1, 1);
        std::cout << "2. DF: Expected: "
            << "{0, 1, 3, 5, 2, 4}"
            << " is: " << vec_to_string(axes_order) << "\n";
        break;
    }
    case SpaceToDepthMode::BLOCKS_FIRST:
    default: { axes_order.insert(axes_order.begin() + spatial_dims + 1, 1);
        std::cout << "2. BF: Expected: "
            << "{0, 3, 5, 1, 2, 4}"
            << " is: " << vec_to_string(axes_order) << "\n";
    }
    }
    flat_node = builder::reorder_axes(flat_node, axes_order);
    Shape squeezed_shape{ n_dim };
    for (int i = 0; i < spatial_dims; ++i)
    {
        squeezed_shape.push_back(data_shape.at(no_spatial_dims + i) / m_blocksize);
    }
    squeezed_shape.insert(squeezed_shape.begin() + 1, c_dim*std::pow(m_blocksize, spatial_dims));
    std::cout << "3. Expected: " << Shape{ n, c_high, h_flat, w_flat } << " is: " << squeezed_shape
        << "\n";
    flat_node = builder::reshape(flat_node, squeezed_shape);

    return NodeVector{flat_node};
}

shared_ptr<Node> op::SpaceToDepth::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<SpaceToDepth>(new_args.at(0), m_mode, m_blocksize);
}

op::SpaceToDepth::SpaceToDepthMode op::SpaceToDepth::mode_from_string(const std::string& mode) const
{
    static const std::map<std::string, SpaceToDepthMode> allowed_values = {
        {"blocks_first", SpaceToDepthMode::BLOCKS_FIRST},
        {"depth_first", SpaceToDepthMode::DEPTH_FIRST}};

    NODE_VALIDATION_CHECK(
        this, allowed_values.count(mode) > 0, "Invalid 'depth_to_space_mode' value passed in.");

    return allowed_values.at(mode);
}
