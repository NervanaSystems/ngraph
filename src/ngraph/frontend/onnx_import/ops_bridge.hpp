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
                explicit UnknownOperator(const std::string& op_type)
                    : ngraph_error{"unknown operator: \"" + op_type + "\""}
                {
                }
            };

            struct UnsupportedVersion : ngraph_error
            {
                explicit UnsupportedVersion(std::int64_t version)
                    : ngraph_error{"unsupported operator set version: " + std::to_string(version)}
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

            static const OperatorSet& get_operator_set(std::int64_t version)
            {
                return instance().get_operator_set_version(version);
            }

        private:
            std::unordered_map<std::string, std::map<std::int64_t, Operator>> m_map;

            OperatorsBridge();

            static const OperatorsBridge& instance()
            {
                static OperatorsBridge instance;
                return instance;
            }

            const OperatorSet& get_operator_set_version_1() const;
            const OperatorSet& get_operator_set_version_2() const;
            const OperatorSet& get_operator_set_version_3() const;
            const OperatorSet& get_operator_set_version_4() const;
            const OperatorSet& get_operator_set_version_5() const;
            const OperatorSet& get_operator_set_version_6() const;
            const OperatorSet& get_operator_set_version_7() const;
            const OperatorSet& get_operator_set_version_8() const;
            const OperatorSet& get_operator_set_version_9() const;
            const OperatorSet& get_operator_set_version(std::int64_t version) const;
        };

    } // namespace onnx_import

} // namespace ngraph
