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

#include <memory>

#include "ngraph/builder/xla_tuple.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/type.hpp"

using namespace std;
using namespace ngraph;

xla::op::Tuple::Tuple(const NodeVector& nodes)
    : Node("Tuple", NodeVector{})
    , m_elements(nodes)
{
}

std::shared_ptr<Node> xla::op::Tuple::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<Tuple>(new_args);
}

const NodeVector& xla::op::Tuple::get_elements() const
{
    return m_elements;
}

size_t xla::op::Tuple::get_tuple_size() const
{
    return m_elements.size();
}

shared_ptr<Node> xla::op::Tuple::get_tuple_element(size_t i)
{
    return m_elements.at(i);
}

shared_ptr<Node> xla::op::get_tuple_element(shared_ptr<Node> node, size_t i)
{
    shared_ptr<xla::op::Tuple> tuple = dynamic_pointer_cast<xla::op::Tuple>(node);
    if (tuple == nullptr)
    {
        throw ngraph_error("get_tuple_element called on a non-tuple");
    }
    return tuple->get_tuple_element(i);
}

namespace
{
    // Add the node to nodes if it's not a Tuple, otherwise add nodes for the elements of the tuple.
    template <typename T>
    void flatten(vector<shared_ptr<T>>& nodes, shared_ptr<Node> node)
    {
        auto xla_tuple = dynamic_pointer_cast<xla::op::Tuple>(node);
        if (xla_tuple == nullptr)
        {
            auto t_node = dynamic_pointer_cast<T>(node);
            if (t_node == nullptr)
            {
                throw ngraph_error("Invalid node type type encountered");
            }
            nodes.push_back(t_node);
        }
        else
        {
            for (auto element : xla_tuple->get_elements())
            {
                flatten<T>(nodes, element);
            }
        }
    }

    // Collect a vector of the non-Tuple nodes that underly nodes
    template <typename T>
    vector<shared_ptr<T>> flatten(const NodeVector& nodes)
    {
        vector<shared_ptr<T>> result;
        for (auto node : nodes)
        {
            flatten<T>(result, node);
        }
        return result;
    }
}

xla::XLAFunction::XLAFunction(const NodeVector& results,
                              const NodeVector& parameters,
                              const string& name)
    : Function(flatten<Node>(results), flatten<ngraph::op::Parameter>(parameters), name)
{
}

xla::XLATuple::XLATuple(const XLAValues& elements)

    : runtime::TensorView(make_shared<descriptor::PrimaryTensorView>(
          make_shared<ngraph::TensorViewType>(element::f32, Shape{}),
          "XLATuple",
          false,
          false,
          false))
    , m_elements(elements)
{
}

const vector<shared_ptr<runtime::TensorView>>& xla::XLATuple::get_elements() const
{
    return m_elements;
}

size_t xla::XLATuple::get_tuple_size() const
{
    return m_elements.size();
}

shared_ptr<runtime::TensorView> xla::XLATuple::get_tuple_element(size_t i) const
{
    return m_elements.at(i);
}

void xla::XLATuple::write(const void* p, size_t tensor_offset, size_t n)
{
    throw ngraph_error("Cannot write to a tuple");
}

void xla::XLATuple::read(void* p, size_t tensor_offset, size_t n) const
{
    throw ngraph_error("Cannot read from a tuple");
}

std::shared_ptr<runtime::TensorView> xla::get_tuple_element(std::shared_ptr<xla::XLATuple> tuple,
                                                            size_t i)
{
    return tuple->get_tuple_element(i);
}

namespace
{
    // Collect the real tensors, expanding the tensors that are really tuples
    void flatten(runtime::TensorViewPtrs& tensors, shared_ptr<runtime::TensorView> tensor)
    {
        auto xla_tuple = dynamic_pointer_cast<xla::XLATuple>(tensor);
        if (xla_tuple == nullptr)
        {
            tensors.push_back(tensor);
        }
        else
        {
            for (auto element : xla_tuple->get_elements())
            {
                flatten(tensors, element);
            }
        }
    }

    // Return a vector of the real tensors underlying a vector of tensors which may contain tuples.
    runtime::TensorViewPtrs flatten(const runtime::TensorViewPtrs& tensors)
    {
        runtime::TensorViewPtrs result;
        for (auto tensor : tensors)
        {
            flatten(result, tensor);
        }
        return result;
    }
}

void xla::call(shared_ptr<runtime::CallFrame> call_frame,
               const runtime::TensorViewPtrs& outputs,
               const runtime::TensorViewPtrs& inputs)
{
    runtime::TensorViewPtrs flat_outputs(flatten(outputs));
    runtime::TensorViewPtrs flat_inputs(flatten(inputs));
    call_frame->tensor_call(flat_outputs, flat_inputs);
}
