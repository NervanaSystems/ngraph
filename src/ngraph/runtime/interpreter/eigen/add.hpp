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

#include "ngraph/runtime/interpreter/call_frame.hpp"
#include "ngraph/runtime/interpreter/eigen/utils.hpp"
#include "ngraph/runtime/interpreter/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            namespace eigen
            {
                template <typename T>
                void AddInstruction(const TensorViewInfo& arg0,
                                    const TensorViewInfo& arg1,
                                    const TensorViewInfo& out)
                {
                    EigenArray1d<T>(out) =
                        EigenArray1d<T>(arg0) + EigenArray1d<T>(arg1);
                }
            }
        }
    }
}
