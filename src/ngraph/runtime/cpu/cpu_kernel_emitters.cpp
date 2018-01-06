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

#include "ngraph/runtime/cpu/cpu_kernel_emitters.hpp"
#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/runtime/cpu/cpu_kernel_utils.hpp"

using namespace ngraph;
using namespace ngraph::runtime::cpu::kernels;

//
// For the reference kernel this is based on, see ngraph/runtime/kernel/concat.hpp.
//
void ngraph::runtime::cpu::kernels::emit_concat(codegen::CodeWriter& writer,
                                                std::string element_type,
                                                const std::vector<std::string> args,
                                                std::string out,
                                                const std::vector<Shape>& in_shapes,
                                                const Shape& out_shape,
                                                size_t concatenation_axis)
{
    size_t concatenation_pos = 0;

    for (size_t i = 0; i < args.size(); i++)
    {
        Coordinate out_start_coord = Coordinate(out_shape.size(), 0);
        out_start_coord[concatenation_axis] = concatenation_pos;

        Coordinate out_end_coord = out_shape;
        out_end_coord[concatenation_axis] = concatenation_pos + in_shapes[i][concatenation_axis];

        CoordinateTransform input_transform(in_shapes[i]);
        CoordinateTransform output_chunk_transform(out_shape, out_start_coord, out_end_coord);

        emit_pointwise_copy(
            writer, element_type, args[i], out, input_transform, output_chunk_transform);

        concatenation_pos += in_shapes[i][concatenation_axis];
    }
}
