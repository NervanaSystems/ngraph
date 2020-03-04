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

// This collection contains one entry for each new op on top of the opset1.

#ifndef NGRAPH_OP
#warning "NGRAPH_OP not defined"
#define NGRAPH_OP(x, y)
#endif

#include "opset1_tbl.hpp"
NGRAPH_OP(Gelu, ngraph::op::v0)
NGRAPH_OP(BatchToSpace, ngraph::op::v1)
NGRAPH_OP(SpaceToBatch, ngraph::op::v1)
