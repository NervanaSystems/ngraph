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
#include <memory>

#include "core/graph.hpp"
#include "core/model.hpp"
#include "core/node.hpp"
#include "ngraph/except.hpp"
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

        std::shared_ptr<Function> import_onnx_model(std::istream& sin, const Weights& weights)
        {
            onnx::ModelProto model_proto;
            if (!model_proto.ParseFromIstream(&sin))
            {
                throw detail::error::stream_parse{sin};
            }
            Model model{model_proto};
            Graph graph{model_proto.graph(), model, weights};
            auto function = std::make_shared<Function>(
                graph.get_ng_outputs(), graph.get_ng_parameters(), graph.get_name());
            for (std::size_t i{0}; i < function->get_output_size(); ++i)
            {
                function->get_output_op(i)->set_name(graph.get_outputs().at(i).get_name());
            }
            return function;
        }

        std::shared_ptr<Function> import_onnx_model(const std::string& path, const Weights& weights)
        {
            std::ifstream ifs{path, std::ios::in | std::ios::binary};
            if (!ifs.is_open())
            {
                throw detail::error::file_open{path};
            }
            return import_onnx_model(ifs, weights);
        }

        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn)
        {
            OperatorsBridge::register_operator(name, version, domain, std::move(fn));
        }

        std::set<std::string> get_supported_operators(std::int64_t version,
                                                      const std::string& domain)
        {
            OperatorSet op_set{OperatorsBridge::get_operator_set(version, domain)};
            std::set<std::string> op_list{};
            for (const auto& op : op_set)
            {
                op_list.emplace(op.first);
            }
            return op_list;
        }

    } // namespace onnx_import

} // namespace ngraph
