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

#include "ngraph/opsets/opset.hpp"
#include "ngraph/ops.hpp"

const ngraph::OpSet& ngraph::get_opset0()
{
    static OpSet opset({
#define NGRAPH_OP(NAME, NAMESPACE) NAMESPACE::NAME::type_info,
#include "ngraph/opsets/opset1_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}

const ngraph::OpSet& ngraph::get_opset1()
{
    static OpSet opset({
#define NGRAPH_OP(NAME, NAMESPACE) NAMESPACE::NAME::type_info,
#include "ngraph/opsets/opset1_tbl.hpp"
#undef NGRAPH_OP
    });
    return opset;
}