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

#include <exception>
#include <ngraph/op/parameter.hpp>
#include <ostream>
#include <vector>

#include "mapping.hpp"
#include "onnx.pb.h"
#include "value_info.hpp"

using namespace ngraph;

onnx_import::ValueInfo::ValueInfo(const onnx::ValueInfoProto& proto, onnx_import::Graph* graph_ptr)
    : m_value_info_proto(proto)
    , m_graph_ptr(graph_ptr)
{
}

std::ostream& onnx_import::operator<<(std::ostream& os, const onnx_import::ValueInfo& wrapper)
{
    std::string name = wrapper.m_value_info_proto.name();
    os << "<ValueInfo: " << name << ">";
    return os;
}

const ngraph::Shape onnx_import::ValueInfo::get_shape() const
{
    onnx::TensorShapeProto tensor_shape_proto = m_value_info_proto.type().tensor_type().shape();
    std::vector<size_t> shape;

    for (const auto& dim : tensor_shape_proto.dim())
        shape.push_back(dim.dim_value());

    return ngraph::Shape(shape);
}

const ngraph::element::Type onnx_import::ValueInfo::get_element_type() const
{
    onnx::TensorProto_DataType onnx_element_type =
        m_value_info_proto.type().tensor_type().elem_type();

    try
    {
        return mapping::onnx_to_ng_type(onnx_element_type);
    }
    catch (std::exception& e)
    {
        throw ngraph::ngraph_error("ValueInfo(" + m_value_info_proto.name() + "): " + e.what());
    }
}

std::shared_ptr<ngraph::op::Parameter> onnx_import::ValueInfo::get_ng_parameter() const
{
    return std::make_shared<op::Parameter>(get_element_type(), get_shape());
}

std::shared_ptr<ngraph::Node> onnx_import::ValueInfo::get_ng_node() const
{
    // @TODO: Add support for values with initializers (constants)
    return get_ng_parameter();
}

bool onnx_import::ValueInfo::has_initializer() const
{
    // @ TODO: Implement
    return false;
}
