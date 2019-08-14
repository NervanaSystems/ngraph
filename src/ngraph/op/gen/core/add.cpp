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

#include "ngraph/op/gen/core/add.hpp"

const std::string ngraph::op::gen::core::Add::type_name = "core.Add";

::std::shared_ptr<::ngraph::Node>
    ngraph::op::gen::core::Add::build(const ::ngraph::OutputVector& source_outputs,
                                      const ::std::vector<const AttributeBase*>& attributes)
{
    return ::std::make_shared<::ngraph::op::gen::core::Add>(source_outputs, attributes);
}
bool ::ngraph::op::gen::core::Add::s_registered =
    ::ngraph::register_gen_op("core.Add", new ::ngraph::op::gen::core::Add::Builder());
