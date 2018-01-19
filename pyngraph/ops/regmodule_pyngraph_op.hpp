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

#pragma once

#include <pybind11/pybind11.h>
#include "pyngraph/ops/abs.hpp"
#include "pyngraph/ops/acos.hpp"
#include "pyngraph/ops/cos.hpp"
#include "pyngraph/ops/add.hpp"
#include "pyngraph/ops/asin.hpp"
#include "pyngraph/ops/broadcast.hpp"
#include "pyngraph/ops/constant.hpp"
#include "pyngraph/ops/convert.hpp"
#include "pyngraph/ops/divide.hpp"
#include "pyngraph/ops/dot.hpp"
#include "pyngraph/ops/exp.hpp"
#include "pyngraph/ops/greater.hpp"
#include "pyngraph/ops/less.hpp"
#include "pyngraph/ops/log.hpp"
#include "pyngraph/ops/maximum.hpp"
#include "pyngraph/ops/minimum.hpp"
#include "pyngraph/ops/multiply.hpp"
#include "pyngraph/ops/negative.hpp"
#include "pyngraph/ops/op.hpp"
#include "pyngraph/ops/one_hot.hpp"
#include "pyngraph/ops/parameter.hpp"
#include "pyngraph/ops/reduce.hpp"
#include "pyngraph/ops/reshape.hpp"
#include "pyngraph/ops/sin.hpp"
#include "pyngraph/ops/subtract.hpp"
#include "pyngraph/ops/sum.hpp"

namespace py = pybind11;

void regmodule_pyngraph_op(py::module m);
