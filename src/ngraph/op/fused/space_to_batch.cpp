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
#include <cmath>
#include <cstddef>
#include <memory>

#include "space_to_batch.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::SpaceToBatch::type_info;

ngraph::op::v1::SpaceToBatch::SpaceToBatch(const ngraph::Output<ngraph::Node> &data,
                                               const ngraph::Output<ngraph::Node> &block_shape,
                                               const ngraph::Output<ngraph::Node> &pads_begin,
                                               const ngraph::Output<ngraph::Node> &pads_end)
                                               : FusedOp({data, block_shape, pads_begin, pads_end})
{
    constructor_validate_and_infer_types();
}


NodeVector op::v1::SpaceToBatch::decompose_op() const {
    auto data = input_value(0);
    auto block = input_value(1);
    auto pads_begin = input_value(2);
    auto pads_end = input_value(3);

    auto data_shape = data.get_shape();
    auto block_shape = block.get_shape();

    NODE_VALIDATION_CHECK(this,
                          (data_shape.size() >= 2),
                          "The input tensor with rank lower than 2 is not supported (input rank: ",
                          data_shape.size(),
                          ")");

    NGRAPH_CHECK(block.get_node_shared_ptr()->is_constant(),
                 "block_shape input node is expected to be a static constant");

    NGRAPH_CHECK(pads_begin.get_node_shared_ptr()->is_constant(),
                 "paddings input node is expected to be a static constant");

    NGRAPH_CHECK(pads_end.get_node_shared_ptr()->is_constant(),
                 "paddings input node is expected to be a static constant");


    const auto block_const = as_type_ptr<op::Constant>(block.get_node_shared_ptr());
    const auto pads_begin_const = as_type_ptr<op::Constant>(pads_begin.get_node_shared_ptr());
    const auto pads_end_const = as_type_ptr<op::Constant>(pads_end.get_node_shared_ptr());

    auto get_in_vec_int64 = [](shared_ptr<op::Constant> in_const, vector<int64_t>& values)
    {
        if(in_const->get_element_type() == element::i32)
        {
            auto tmp = in_const->get_vector<int32_t>();
            values.insert(values.begin(), tmp.begin(), tmp.end());
        }
        else
        {
            values = in_const->get_vector<int64_t>();
        }
    };

    vector<int64_t> block_values, pads_begin_values, pads_end_values;
    get_in_vec_int64(block_const, block_values);
    get_in_vec_int64(pads_begin_const, pads_begin_values);
    get_in_vec_int64(pads_end_const, pads_end_values);

    auto out = make_shared<op::v1::Pad>(data, pads_begin_const, pads_end_const, PadMode::CONSTANT);
    auto out_shape = out->get_shape();

    // First we have to disperse the data from spatial dimensions, then
    // rearrange them so as appropriate chunks of data where close to their
    // destination place. Finally squeeze data from respective dimensions.
    Shape dispersed_shape{out_shape.at(0)};
    for (size_t i = 1; i < block_values.size(); ++i)
    {
        dispersed_shape.push_back(out_shape.at(i) / block_values.at(i));
        dispersed_shape.push_back(block_values.at(i));
    }
    auto flat_node = builder::opset1::reshape(out, dispersed_shape);

    vector<size_t> axes_order;
    for (size_t i = 0, j = 2; i < block_values.size()-1; ++i, j += 2)
    {
        axes_order.push_back(j);
    }
    axes_order.push_back(0);
    for (size_t i = 0, j = 1; i < block_values.size()-1; ++i, j += 2)
    {
        axes_order.push_back(j);
    }

    flat_node = builder::opset1::reorder_axes(flat_node, axes_order);
    Shape squeezed_shape;
    int64_t prod = 1;
    for (auto &el : block_values)
    {
        prod *= el;
    }

    squeezed_shape.push_back(out_shape.at(0) * prod);
    for (size_t i = 1; i < block_values.size(); ++i)
    {
        squeezed_shape.push_back(out_shape.at(i) / block_values.at(i));
    }
    flat_node = builder::opset1::reshape(flat_node, squeezed_shape);

    return NodeVector{flat_node};
}

std::shared_ptr<Node> ngraph::op::v1::SpaceToBatch::copy_with_new_args(const ngraph::NodeVector &new_args) const {
    if (new_args.size() != 4)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<SpaceToBatch>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ngraph::op::v1::SpaceToBatch::visit_attributes(ngraph::AttributeVisitor &visitor) {
    return true;
}
