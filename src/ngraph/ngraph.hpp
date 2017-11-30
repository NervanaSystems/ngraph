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

#ifdef IN_NGRAPH_LIBRARY
#error("ngraph.hpp is for external use only")
#endif

/// @namespace ngraph
/// @brief The Intel Nervana Graph C++ API.

/// @namespace ngraph::descriptor
/// @brief Descriptors are compile-time representations of objects that will appear at run-time.

/// @namespace ngraph::descriptor::layout
/// @brief Layout descriptors describe how tensor views are implemented.

/// @namespace ngraph::op
/// @brief Ops used in graph-building.

/// @namespace ngraph::runtime
/// @brief The objects used for executing the graph.

/// @namespace ngraph::builder
/// @brief Convenience functions that create addional graph nodes to implement commonly-used
///        recipes, for example auto-broadcast.

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/numpy_transpose.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/common.hpp"
#include "ngraph/descriptor/buffer.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/descriptor/tuple.hpp"
#include "ngraph/descriptor/value.hpp"
#include "ngraph/except.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/acos.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/asin.hpp"
#include "ngraph/ops/atan.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/ceiling.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/cos.hpp"
#include "ngraph/ops/cosh.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/ops/floor.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_tuple_element.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/minimum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/remainder.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/sign.hpp"
#include "ngraph/ops/sin.hpp"
#include "ngraph/ops/sinh.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sqrt.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/tan.hpp"
#include "ngraph/ops/tanh.hpp"
#include "ngraph/ops/tuple.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/ngvm/ngvm_backend.hpp"
#include "ngraph/runtime/ngvm/ngvm_manager.hpp"
#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/runtime/tuple.hpp"
#include "ngraph/runtime/value.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/types/element_type.hpp"
#include "ngraph/types/type.hpp"
