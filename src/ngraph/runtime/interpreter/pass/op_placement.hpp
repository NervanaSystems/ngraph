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

#pragma once

#include <memory>
#include <set>

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            namespace pass
            {
                class OpPlacement;
            }
        }
    }
}

class ngraph::runtime::interpreter::pass::OpPlacement : public ngraph::pass::FunctionPass
{
public:
    OpPlacement(std::set<std::string> unsupported_ops);

    enum class DeviceSupport
    {
        UNKNOWN,
        SUPPORTED,
        UNSUPPORTED
    };

private:
    virtual bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
    void assign_placement(std::shared_ptr<ngraph::Node> node);
    DeviceSupport is_supported_on_device(std::shared_ptr<ngraph::Node> node);
    std::set<std::string> m_unsupported_ops;
};
