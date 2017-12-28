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

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/common.hpp"

using namespace std;
namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernels
            {
                void emit_concat(codegen::CodeWriter& writer,
                                 std::string element_type,
                                 const std::vector<std::string> args,
                                 std::string out,
                                 const std::vector<Shape>& in_shapes,
                                 const Shape& out_shape,
                                 size_t concatenation_axis);

                void emit_replace_slice(codegen::CodeWriter& writer,
                                        std::string element_type,
                                        std::string arg0, // replacement context
                                        std::string arg1, // replacement value
                                        std::string out,
                                        const Shape& arg1_shape,
                                        const Shape& out_shape,
                                        const Coordinate& lower_bounds,
                                        const Coordinate& upper_bounds,
                                        const Strides& strides);
            }
        }
    }
}