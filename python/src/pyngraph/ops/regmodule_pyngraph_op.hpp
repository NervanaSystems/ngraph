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

#include <pybind11/pybind11.h>
#include "pyngraph/ops/allreduce.hpp"
#include "pyngraph/ops/argmax.hpp"
#include "pyngraph/ops/argmin.hpp"
#include "pyngraph/ops/avg_pool.hpp"
#include "pyngraph/ops/batch_norm.hpp"
#include "pyngraph/ops/broadcast.hpp"
#include "pyngraph/ops/broadcast_distributed.hpp"
#include "pyngraph/ops/constant.hpp"
#include "pyngraph/ops/convert.hpp"
#include "pyngraph/ops/convolution.hpp"
#include "pyngraph/ops/dequantize.hpp"
#include "pyngraph/ops/dot.hpp"
#include "pyngraph/ops/fused/depth_to_space.hpp"
#include "pyngraph/ops/fused/gelu.hpp"
#include "pyngraph/ops/fused/gemm.hpp"
#include "pyngraph/ops/fused/grn.hpp"
#include "pyngraph/ops/fused/group_conv.hpp"
#include "pyngraph/ops/fused/hard_sigmoid.hpp"
#include "pyngraph/ops/fused/mvn.hpp"
#include "pyngraph/ops/fused/rnn_cell.hpp"
#include "pyngraph/ops/fused/scale_shift.hpp"
#include "pyngraph/ops/fused/shuffle_channels.hpp"
#include "pyngraph/ops/fused/space_to_depth.hpp"
#include "pyngraph/ops/fused/unsqueeze.hpp"
#include "pyngraph/ops/get_output_element.hpp"
#include "pyngraph/ops/max.hpp"
#include "pyngraph/ops/max_pool.hpp"
#include "pyngraph/ops/maximum.hpp"
#include "pyngraph/ops/min.hpp"
#include "pyngraph/ops/parameter.hpp"
#include "pyngraph/ops/passthrough.hpp"
#include "pyngraph/ops/product.hpp"
#include "pyngraph/ops/quantize.hpp"
#include "pyngraph/ops/quantized_convolution.hpp"
#include "pyngraph/ops/quantized_dot.hpp"
#include "pyngraph/ops/replace_slice.hpp"
#include "pyngraph/ops/result.hpp"
#include "pyngraph/ops/slice.hpp"
#include "pyngraph/ops/softmax.hpp"

namespace py = pybind11;

void regmodule_pyngraph_op(py::module m);
