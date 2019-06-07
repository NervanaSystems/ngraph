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

#pragma once

#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    /// \brief Serialize a Function to a json string
    /// \param func The Function to serialize
    /// \param indent If 0 then there is no formatting applied and the resulting string is the
    ///    most compact representation. If non-zero then the json string is formatted with the
    ///    indent level specified.
    std::string serialize(std::shared_ptr<ngraph::Function> func, size_t indent = 0);

    /// \brief Serialize a Function to a json file
    /// \param path The path to the output file
    /// \param func The Function to serialize
    /// \param indent If 0 then there is no formatting applied and the resulting string is the
    ///    most compact representation. If non-zero then the json string is formatted with the
    ///    indent level specified.
    void serialize(const std::string& path,
                   std::shared_ptr<ngraph::Function> func,
                   size_t indent = 0);

    /// \brief Serialize a Function to a json stream
    /// \param out The output stream to which the data is serialized.
    /// \param func The Function to serialize
    /// \param indent If 0 then there is no formatting applied and the json is the
    ///    most compact representation. If non-zero then the json is formatted with the
    ///    indent level specified.
    void serialize(std::ostream& out, std::shared_ptr<ngraph::Function> func, size_t indent = 0);

    /// \brief Deserialize a Function
    /// \param in An isteam to the input data
    std::shared_ptr<ngraph::Function> deserialize(std::istream& in);

    /// \brief Deserialize a Function
    /// \param str The json formatted string to deseriailze.
    std::shared_ptr<ngraph::Function> deserialize(const std::string& str);

    /// \brief If enabled adds output shapes to the serialized graph
    /// \param enable Set to true to enable or false otherwise
    ///
    /// Option may be enabled by setting the environment variable NGRAPH_SERIALIZER_OUTPUT_SHAPES
    void set_serialize_output_shapes(bool enable);
}

#ifndef NGRAPH_JSON_ENABLE
// Rather than making every reference to the serializer conditionally compile here we just
// provide some null stubs to resolve link issues
// The `inline` is so we don't get multiple definitions of function
std::string inline ngraph::serialize(std::shared_ptr<ngraph::Function> func, size_t indent)
{
    return "";
}

void inline ngraph::serialize(const std::string& path,
                              std::shared_ptr<ngraph::Function> func,
                              size_t indent)
{
    throw std::runtime_error("serializer disabled in build");
}

void inline ngraph::serialize(std::ostream& out,
                              std::shared_ptr<ngraph::Function> func,
                              size_t indent)
{
    throw std::runtime_error("serializer disabled in build");
}

std::shared_ptr<ngraph::Function> inline ngraph::deserialize(std::istream& in)
{
    throw std::runtime_error("serializer disabled in build");
}

std::shared_ptr<ngraph::Function> inline ngraph::deserialize(const std::string& str)
{
    throw std::runtime_error("serializer disabled in build");
}

void inline ngraph::set_serialize_output_shapes(bool enable)
{
    throw std::runtime_error("serializer disabled in build");
}
#endif
