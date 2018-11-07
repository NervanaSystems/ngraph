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

#include <fstream>

#include "ngraph/except.hpp"

#include "core/graph.hpp"
#include "core/model.hpp"
#include "core/node.hpp"

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
                        : ngraph_error{"failure opening file:" + path}
                    {
                    }
                };

                struct stream_parse : ngraph_error
                {
                    explicit stream_parse(std::istream&)
                        : ngraph_error{"failure parsing data from the stream"}
                    {
                    }
                };

            } // namespace error
        }     // namespace detail

        std::vector<std::shared_ptr<Function>> load_onnx_model(std::istream& sin,
                                                               const Weights& weights)
        {
            onnx::ModelProto model_proto;
            if (!model_proto.ParseFromIstream(&sin))
            {
                throw detail::error::stream_parse{sin};
            }
            std::vector<std::shared_ptr<Function>> output_functions;
            Model model{model_proto};
            Graph graph{model_proto.graph(), model, weights};
            for (const auto& output : graph.get_outputs())
            {
                output_functions.emplace_back(std::make_shared<Function>(
                    graph.get_ng_node_from_cache(output.get_name()), graph.get_ng_parameters()));
            }
            return output_functions;
        }

        std::vector<std::shared_ptr<Function>> load_onnx_model(const std::string& path,
                                                               const Weights& weights)
        {
            std::ifstream ifs{path, std::ios::in | std::ios::binary};
            if (!ifs.is_open())
            {
                throw detail::error::file_open{path};
            }
            return load_onnx_model(ifs, weights);
        }

        std::shared_ptr<Function> import_onnx_function(std::istream& sin, const Weights& weights)
        {
            return load_onnx_model(sin, weights).front();
        }

        std::shared_ptr<Function> import_onnx_function(const std::string& path,
                                                       const Weights& weights)
        {
            return load_onnx_model(path, weights).front();
        }

        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn)
        {
            OperatorsBridge::register_operator(name, version, domain, std::move(fn));
        }

    } // namespace onnx_import

} // namespace ngraph
