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
#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernels
            {
                std::vector<std::string> emit_multi_indices(CoordinateTransform trans,
                                                            std::vector<std::string> index_vars);
                std::string emit_linear_index(CoordinateTransform trans,
                                              std::vector<std::string> index_vars);
                std::string
                    start_index_loop(std::string index_var, size_t start, size_t end, bool omp);
                std::string end_index_loop(std::string index_var);
                void emit_pointwise_copy(codegen::CodeWriter& writer,
                                         std::string source_buffer,
                                         std::string dest_buffer,
                                         CoordinateTransform source_trans,
                                         CoordinateTransform dest_trans);
            }
        }
    }
}
