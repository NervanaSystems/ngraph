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

#include "space_to_depth.hpp"
#include "exceptions.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector space_to_depth(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    const Shape& data_shape = data->get_shape();

                    std::int64_t block_size{node.get_attribute_value<std::int64_t>("blocksize")};

                    // Set default values to each dimension to be able to work with both 3D or 4D data.
                    std::size_t n{1}, c{1}, h{1}, w{1};
                    ASSERT_VALID_ARGUMENT(node, (data_shape.size() == 3 || data_shape.size() == 4))
                        << "The provided tensor shape: " << data_shape << " is not supported.";
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
                    ASSERT_VALID_ARGUMENT(
                        node, (h % block_size == 0 && w % block_size == 0 && block_size > 0))
                        << "The width and height axes size must be a multiple of squared block_size"
                           " attribute value";
                    std::size_t bs = static_cast<std::size_t>(block_size);
                    std::size_t w_flat = w / bs;
                    std::size_t h_flat = h / bs;
                    std::size_t c_high = c * bs * bs;

                    // First we have to disperse the data from height and width channels, then
                    // rearrange them so as appropriate chunks of data where close to their
                    // destination place. Finally squeeze data from respective dimensions.
                    std::shared_ptr<ngraph::Node> flat_node =
                        reshape::reshape(data, ngraph::Shape{n, c, h_flat, bs, w_flat, bs});
                    flat_node = reshape::reorder_axes(flat_node, {0, 3, 5, 1, 2, 4});
                    return {reshape::reshape(flat_node, ngraph::Shape{n, c_high, h_flat, w_flat})};
                }
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
