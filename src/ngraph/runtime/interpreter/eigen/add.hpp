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

#include "ngraph/log.hpp"
#include "ngraph/runtime/interpreter/eigen/eigen_utils.hpp"
#include "ngraph/runtime/interpreter/int_tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            namespace eigen
            {
                template <typename T>
                void AddInstruction(std::shared_ptr<INT_TensorView> arg0,
                                    std::shared_ptr<INT_TensorView> arg1,
                                    std::shared_ptr<INT_TensorView> out)
                {
                    size_t element_count = out->get_element_count();
                    T* data0 = reinterpret_cast<T*>(arg0->get_data_ptr());
                    T* data1 = reinterpret_cast<T*>(arg1->get_data_ptr());
                    T* out0 = reinterpret_cast<T*>(out->get_data_ptr());
                    for (size_t i = 0; i < element_count; i++)
                    {
                        NGRAPH_INFO << data0[i] << ", " << data1[i];
                        out0[i] = data0[i] + data1[i];
                    }
                }
            }
        }
    }
}
