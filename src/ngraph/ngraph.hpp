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

//
// The public API for ngraph++
//

#pragma once

#include "ngraph/common.hpp"
#include "ngraph/descriptor/buffer.hpp"
#include "ngraph/descriptor/call_frame.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/descriptor/tensor_view_layout.hpp"
#include "ngraph/element_type.hpp"
#include "ngraph/except.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/ceiling.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/floor.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/tuple.hpp"
#include "ngraph/runtime/eigen/add.hpp"
#include "ngraph/runtime/eigen/multiply.hpp"
#include "ngraph/runtime/eigen/return.hpp"
#include "ngraph/runtime/eigen/tensor_view.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/function.hpp"
#include "ngraph/runtime/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type.hpp"
