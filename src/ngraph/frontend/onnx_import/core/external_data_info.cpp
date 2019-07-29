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

#include <fstream>
#include <iostream>
#include <onnx/onnx_pb.h>

#include "external_data_info.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        ExternalDataInfo::ExternalDataInfo(const onnx::TensorProto& tensor)
        {
            for (const auto& entry : tensor.external_data())
            {
                if (entry.key() == "location")
                    m_data_location = entry.value();
                if (entry.key() == "offset")
                    m_offset = std::stoi(entry.value());
                if (entry.key() == "length")
                    m_data_lenght = std::stoi(entry.value());
                if (entry.key() == "checksum")
                    m_sha1_digest = std::stoi(entry.value());
            }
        }

        std::string ExternalDataInfo::load_external_data() const
        {
            std::ifstream external_data_stream(m_data_location,
                                               std::ios::binary | std::ios::in | std::ios::ate);
            if (external_data_stream.fail())
                throw invalid_external_data{*this};
            std::streamsize data_byte_lenght = external_data_stream.tellg();
            std::cout << "data_byte_lenght: " << data_byte_lenght << "\n";
            external_data_stream.seekg(0, std::ios::beg);
            //TODO OFFSETS, CHECKSUM HANDLING
            std::string read_data;
            read_data.resize(data_byte_lenght);
            external_data_stream.read(&read_data[0], data_byte_lenght);
            external_data_stream.close();
            return read_data;
        }
    }
}