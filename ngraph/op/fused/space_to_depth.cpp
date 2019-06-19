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
#include <cstdint>
#include <memory>

#include "ngraph/builder/reshape.hpp"
#include "ngraph/shape.hpp"
#include "space_to_depth.hpp"

using namespace std;
using namespace ngraph;

op::SpaceToDepth::SpaceToDepth(const shared_ptr<Node>& data, const size_t block_size)
    : FusedOp("SpaceToDepth", {data})
    , m_blocksize(block_size)
{
    constructor_validate_and_infer_types();
}

NodeVector op::SpaceToDepth::decompose_op() const
{
    auto data = get_argument(0);
    const Shape& data_shape = data->get_shape();

    // Set default values to each dimension to be able to work with both 3D or 4D data.
    size_t n{1}, c{1}, h{1}, w{1};

    NGRAPH_CHECK((data_shape.size() == 3 || data_shape.size() == 4),
                 "The provided tensor shape: ",
                 data_shape,
                 " is not supported.");

    // Assume NCHW data layout
    if (data_shape.size() == 4)
    {
        n = data_shape.at(0);
        c = data_shape.at(1);
        h = data_shape.at(2);
        w = data_shape.at(3);
    }
    // Without batch.
    else if (data_shape.size() == 3)
    {
        c = data_shape.at(0);
        h = data_shape.at(1);
        w = data_shape.at(2);
    }

    NGRAPH_CHECK((h % m_blocksize == 0 && w % m_blocksize == 0 && m_blocksize > 0),
                 "SpaceToDepth: The width and height axes size must be a multiple of ",
                 "squared block_size attribute value");

    size_t bs = static_cast<size_t>(m_blocksize);
    size_t w_flat = w / bs;
    size_t h_flat = h / bs;
    size_t c_high = c * bs * bs;

    // First we have to disperse the data from height and width channels, then
    // rearrange them so as appropriate chunks of data where close to their
    // destination place. Finally squeeze data from respective dimensions.
    shared_ptr<Node> flat_node = builder::reshape(data, Shape{n, c, h_flat, bs, w_flat, bs});
    flat_node = builder::reorder_axes(flat_node, {0, 3, 5, 1, 2, 4});
    return NodeVector{builder::reshape(flat_node, Shape{n, c_high, h_flat, w_flat})};
}

shared_ptr<Node> op::SpaceToDepth::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<SpaceToDepth>(new_args.at(0), m_blocksize);
}
