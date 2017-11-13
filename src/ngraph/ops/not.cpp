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

#include "ngraph/ops/not.hpp"
#include "ngraph/ops/op.hpp"

using namespace ngraph;
using namespace ngraph::op;

op::Not::Not(const std::shared_ptr<Node>& arg)
    : UnaryElementwise(
          "Not",
          [](const ngraph::element::Type& arg_element_type) -> const ngraph::element::Type& {
              if (arg_element_type != element::Bool::element_type())
              {
                  throw ngraph_error(
                      "Operands for logical operators must have boolean element "
                      "type");
              }

              return arg_element_type;
          },
          arg)
{
}
