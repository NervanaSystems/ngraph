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
#include <ops.hpp>

#include "batch_to_space.hpp"
#include "space_to_batch.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::BatchToSpace::type_info;

ngraph::op::v1::BatchToSpace::BatchToSpace(const ngraph::Output<ngraph::Node> &data,
                                           const ngraph::Output<ngraph::Node> &block_shape,
                                           const ngraph::Output<ngraph::Node> &crops_begin,
                                           const ngraph::Output<ngraph::Node> &crops_end)
                                           : FusedOp({data, block_shape, crops_begin, crops_end})
{
    constructor_validate_and_infer_types();
}


NodeVector op::v1::BatchToSpace::decompose_op() const {
    auto data = input_value(0);
    auto block = input_value(1);
    auto crops_begin = input_value(2);
    auto crops_end = input_value(3);

    auto data_shape = data.get_shape();
    auto block_shape = block.get_shape();
    NODE_VALIDATION_CHECK(this,
                          (data_shape.size() >= 2),
                          "The input tensor with rank lower than 2 is not supported (input rank: ",
                          data_shape.size(),
                          ")");

    NGRAPH_CHECK(block.get_node_shared_ptr()->is_constant(),
                 "block_shape input node is expected to be a static constant");

    NGRAPH_CHECK(crops_begin.get_node_shared_ptr()->is_constant(),
                 "paddings input node is expected to be a static constant");

    NGRAPH_CHECK(crops_end.get_node_shared_ptr()->is_constant(),
                 "paddings input node is expected to be a static constant");


    const auto block_const = as_type_ptr<op::Constant>(block.get_node_shared_ptr());
    const auto crops_begin_const = as_type_ptr<op::Constant>(crops_begin.get_node_shared_ptr());
    const auto crops_end_const = as_type_ptr<op::Constant>(crops_end.get_node_shared_ptr());

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

    vector<int64_t> block_values, crops_begin_values, crops_end_values;
    get_in_vec_int64(block_const, block_values);
    get_in_vec_int64(crops_begin_const, crops_begin_values);
    get_in_vec_int64(crops_end_const, crops_end_values);

    // temporary workaround
    if(data_shape.size() == 4) {
        data = builder::opset1::reorder_axes(data, {0, 2, 3, 1});
        data_shape = data.get_shape();
    }

    // First we have to disperse the data from depth channel, then rearrange them
    // so as appropriate chunks of data where close to their destination place.
    // Finally squeeze data from respective dimensions.
    Shape dispersed_shape;
    int64_t prod = 1;
    for (auto &el : block_values)
    {
        prod *= el;
    }

    dispersed_shape.insert(dispersed_shape.begin(), block_values.begin() + 1, block_values.end());
    dispersed_shape.push_back(data_shape.at(0) / prod);
    for (size_t i = 1; i < data_shape.size(); ++i)
    {
        dispersed_shape.push_back(data_shape.at(i));
    }
    auto flat_node = builder::opset1::reshape(data, dispersed_shape);

    // calculate axes to transpose
    // [0, 3, 5, ..., spatial_dims + (spatial_dims + 1), 2, 4, ..., K + K])
    vector<size_t> axes_order{block_values.size()-1};
    for (size_t i = 0; i < block_values.size() - 1; ++i)
    {
        axes_order.push_back(i);
        axes_order.push_back(i + block_values.size());
    }
    flat_node = builder::opset1::reorder_axes(flat_node, axes_order);

    Shape squeezed_shape{data_shape.at(0) / prod};
    for (size_t i = 1; i < block_values.size(); ++i)
    {
        squeezed_shape.push_back(data_shape.at(i) * block_values.at(i));
    }
    flat_node = builder::opset1::reshape(flat_node, squeezed_shape);

    vector<int64_t> upperbounds_values;
    auto flat_node_shape = flat_node->get_shape();
    for(size_t i = 0; i < flat_node_shape.size();++i)
    {
        upperbounds_values.push_back(flat_node_shape.at(i) - crops_end_values.at(i));
    }
    const auto upperbounds = op::Constant::create(crops_end.get_element_type(), Shape{upperbounds_values.size()}, upperbounds_values);

    vector<int64_t> begin_mask(data_shape.size(), 0);
    vector<int64_t> end_mask(data_shape.size(), 0);
    flat_node = make_shared<op::v1::StridedSlice>(flat_node, crops_begin_const, upperbounds, begin_mask, end_mask);
    if(data_shape.size() == 4) {
        flat_node = builder::opset1::reorder_axes(flat_node, {0, 3, 1, 2});
    }
    return NodeVector{flat_node};
}

std::shared_ptr<ngraph::Node> ngraph::op::v1::BatchToSpace::copy_with_new_args(const ngraph::NodeVector &new_args) const {
    if (new_args.size() != 4)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<BatchToSpace>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ngraph::op::v1::BatchToSpace::visit_attributes(ngraph::AttributeVisitor &visitor) {
    return true;
}