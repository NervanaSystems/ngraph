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

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>

#include "ngraph/except.hpp"

#include "core/operator_set.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            struct UnknownOperator : ngraph_error
            {
                UnknownOperator(const std::string& name, const std::string& domain)
                    : ngraph_error{(domain.empty() ? "" : domain + ".") + name}
                {
                }
            };

            struct UnknownDomain : ngraph_error
            {
                explicit UnknownDomain(const std::string& domain)
                    : ngraph_error{domain}
                {
                }
            };

            struct UnsupportedVersion : ngraph_error
            {
                UnsupportedVersion(const std::string& name,
                                   std::int64_t version,
                                   const std::string& domain)
                    : ngraph_error{(domain.empty() ? "" : domain + ".") + name + ":" +
                                   std::to_string(version)}
                {
                }
            };

        } // namespace error

        class OperatorsBridge
        {
        public:
            OperatorsBridge(const OperatorsBridge&) = delete;
            OperatorsBridge& operator=(const OperatorsBridge&) = delete;
            OperatorsBridge(OperatorsBridge&&) = delete;
            OperatorsBridge& operator=(OperatorsBridge&&) = delete;

            static OperatorSet get_operator_set(std::int64_t version, const std::string& domain)
            {
                return instance()._get_operator_set(version, domain);
            }

            static void register_operator(const std::string& name,
                                          std::int64_t version,
                                          const std::string& domain,
                                          Operator fn)
            {
                instance()._register_operator(name, version, domain, std::move(fn));
            }

        private:
            std::unordered_map<std::string,
                               std::unordered_map<std::string, std::map<std::int64_t, Operator>>>
                m_map;

            OperatorsBridge();

            static OperatorsBridge& instance()
            {
                static OperatorsBridge instance;
                return instance;
            }

            void _register_operator(const std::string& name,
                                    std::int64_t version,
                                    const std::string& domain,
                                    Operator fn);
            OperatorSet _get_operator_set(std::int64_t version, const std::string& domain);
        };

    } // namespace onnx_import

} // namespace ngraph
