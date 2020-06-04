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

#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <memory>

#include "core/graph.hpp"
#include "core/model.hpp"
#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
#include "onnx.hpp"
#include "ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            namespace error
            {
                struct file_open : ngraph_error
                {
                    explicit file_open(const std::string& path)
                        : ngraph_error{"Failure opening file: " + path}
                    {
                    }
                };

                struct stream_parse : ngraph_error
                {
                    explicit stream_parse(std::istream&)
                        : ngraph_error{"Failure parsing data from the provided input stream"}
                    {
                    }
                };
            }
        }

        std::shared_ptr<Function> import_onnx_model(std::istream& stream)
        {
            NGRAPH_INFO;
            ONNX_NAMESPACE::ModelProto model_proto;
            // Try parsing input as a binary protobuf message
            if (!model_proto.ParseFromIstream(&stream))
            {
                // Rewind to the beginning and clear stream state.
                stream.clear();
                stream.seekg(0);
                google::protobuf::io::IstreamInputStream iistream(&stream);
                // Try parsing input as a prototxt message
                if (!google::protobuf::TextFormat::Parse(&iistream, &model_proto))
                {
                    throw detail::error::stream_parse{stream};
                }
            }

            Model model{model_proto};
            Graph graph{model_proto.graph(), model};
            NGRAPH_INFO;
            auto a = graph.get_ng_outputs();
            NGRAPH_INFO;
            auto b = graph.get_ng_parameters();
            NGRAPH_INFO;
            auto function = std::make_shared<Function>(
                a, b, graph.get_name());
            NGRAPH_INFO;
            for (std::size_t i{0}; i < function->get_output_size(); ++i)
            {
            NGRAPH_INFO;
                function->get_output_op(i)->set_friendly_name(graph.get_outputs().at(i).get_name());
            NGRAPH_INFO;
            }
            NGRAPH_INFO;
            return function;
        }

        std::shared_ptr<Function> import_onnx_model(const std::string& file_path)
        {
            NGRAPH_INFO;
            std::ifstream ifs{file_path, std::ios::in | std::ios::binary};
            if (!ifs.is_open())
            {
                throw detail::error::file_open{file_path};
            }
            return import_onnx_model(ifs);
        }

        std::set<std::string> get_supported_operators(std::int64_t version,
                                                      const std::string& domain)
        {
            OperatorSet op_set{
                OperatorsBridge::get_operator_set(domain == "ai.onnx" ? "" : domain, version)};
            std::set<std::string> op_list{};
            for (const auto& op : op_set)
            {
                op_list.emplace(op.first);
            }
            return op_list;
        }

        bool is_operator_supported(const std::string& op_name,
                                   std::int64_t version,
                                   const std::string& domain)
        {
            return OperatorsBridge::is_operator_registered(
                op_name, version, domain == "ai.onnx" ? "" : domain);
        }
    }
}
