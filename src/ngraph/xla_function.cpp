// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>

#include "ngraph/log.hpp"
#include "ngraph/ops/xla_tuple.hpp"
#include "ngraph/util.hpp"
#include "ngraph/xla_function.hpp"

using namespace std;
using namespace ngraph;

XLAFunction::XLAFunction(const std::shared_ptr<Node>& result,
                         const std::shared_ptr<const ValueType>& result_type,
                         const std::vector<std::shared_ptr<op::Parameter>>& parameters,
                         const std::string& name)
    : Function(Nodes{result},
               std::vector<std::shared_ptr<const ValueType>>{result_type},
               parameters,
               name){};
