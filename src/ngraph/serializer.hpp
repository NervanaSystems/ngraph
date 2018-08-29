//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

    /// \brief Serialize a Function to as a json file
    /// \param path The path to the output file
    /// \param func The Function to serialize
    /// \param indent If 0 then there is no formatting applied and the resulting string is the
    ///    most compact representation. If non-zero then the json string is formatted with the
    ///    indent level specified.
    void serialize(const std::string& path,
                   std::shared_ptr<ngraph::Function> func,
                   size_t indent = 0);

    /// \brief Serialize a Function to a CPIO file with all constant data stored as binary
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
}
