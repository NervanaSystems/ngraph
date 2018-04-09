/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <pybind11/pybind11.h>
#include "pyngraph/ops/abs.hpp"
#include "pyngraph/ops/acos.hpp"
#include "pyngraph/ops/add.hpp"
#include "pyngraph/ops/asin.hpp"
#include "pyngraph/ops/atan.hpp"
#include "pyngraph/ops/avg_pool.hpp"
#include "pyngraph/ops/broadcast.hpp"
#include "pyngraph/ops/ceiling.hpp"
#include "pyngraph/ops/concat.hpp"
#include "pyngraph/ops/constant.hpp"
#include "pyngraph/ops/convert.hpp"
#include "pyngraph/ops/convolution.hpp"
#include "pyngraph/ops/cos.hpp"
#include "pyngraph/ops/cosh.hpp"
#include "pyngraph/ops/divide.hpp"
#include "pyngraph/ops/dot.hpp"
#include "pyngraph/ops/equal.hpp"
#include "pyngraph/ops/exp.hpp"
#include "pyngraph/ops/floor.hpp"
#include "pyngraph/ops/greater.hpp"
#include "pyngraph/ops/greater_eq.hpp"
#include "pyngraph/ops/less.hpp"
#include "pyngraph/ops/less_eq.hpp"
#include "pyngraph/ops/log.hpp"
#include "pyngraph/ops/max_pool.hpp"
#include "pyngraph/ops/maximum.hpp"
#include "pyngraph/ops/minimum.hpp"
#include "pyngraph/ops/multiply.hpp"
#include "pyngraph/ops/negative.hpp"
#include "pyngraph/ops/not.hpp"
#include "pyngraph/ops/not_equal.hpp"
// #include "pyngraph/ops/op.hpp"
#include "pyngraph/ops/allreduce.hpp"
#include "pyngraph/ops/batch_norm.hpp"
#include "pyngraph/ops/function_call.hpp"
#include "pyngraph/ops/get_output_element.hpp"
#include "pyngraph/ops/max.hpp"
#include "pyngraph/ops/min.hpp"
#include "pyngraph/ops/one_hot.hpp"
#include "pyngraph/ops/pad.hpp"
#include "pyngraph/ops/parameter.hpp"
#include "pyngraph/ops/parameter_vector.hpp"
#include "pyngraph/ops/power.hpp"
#include "pyngraph/ops/product.hpp"
#include "pyngraph/ops/reduce.hpp"
#include "pyngraph/ops/relu.hpp"
#include "pyngraph/ops/replace_slice.hpp"
#include "pyngraph/ops/reshape.hpp"
#include "pyngraph/ops/reverse.hpp"
#include "pyngraph/ops/select.hpp"
#include "pyngraph/ops/sign.hpp"
#include "pyngraph/ops/sin.hpp"
#include "pyngraph/ops/sinh.hpp"
#include "pyngraph/ops/slice.hpp"
#include "pyngraph/ops/softmax.hpp"
#include "pyngraph/ops/sqrt.hpp"
#include "pyngraph/ops/subtract.hpp"
#include "pyngraph/ops/sum.hpp"
#include "pyngraph/ops/tan.hpp"
#include "pyngraph/ops/tanh.hpp"

namespace py = pybind11;

void regmodule_pyngraph_op(py::module m);
