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

#include "ngraph/runtime/cpu/cpu_kernel_utils.hpp"
#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/common.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace ngraph::runtime::cpu::kernels;

//
// Given a coordinate transform and a vector of index expressions relative to
// the target coordinate space, produces the strings needed to index into the
// source coordinate space if it is represented as a multidimensional array.
//
// For example,
//
//    trans has stride (2,2,2), axis order (2,0,1), and start offsets (3,4,5)
//
//    index_vars are "i", "j", "k"
//
// this will produce:
//
//    {"((k) * 2 + 5)", "((i) * 2 + 3)", "((j) * 2 + 4)"}
//
//
std::vector<std::string>
    ngraph::runtime::cpu::kernels::emit_multi_indices(CoordinateTransform trans,
                                                      std::vector<std::string> index_vars)
{
    std::vector<std::string> result;

    for (size_t i = 0; i < index_vars.size(); i++)
    {
        std::string index_var = index_vars[trans.get_source_axis_order()[i]];
        size_t source_stride = trans.get_source_strides()[i];
        size_t source_start = trans.get_source_start_corner()[i];
        std::stringstream ss;

        if (source_stride == 1 && source_start == 0)
        {
            ss << index_var;
        }
        else if (source_stride == 1)
        {
            ss << "((" << index_var << ") + " << source_start << ")";
        }
        else if (source_start == 0)
        {
            ss << "(" << source_stride << " * (" << index_var << "))";
        }
        else
        {
            ss << "(" << source_stride << " * (" << index_var << ") + " << source_start << ")";
        }

        result.push_back(ss.str());
    }

    return result;
}

//
// Given a coordinate transform and a vector of index expressions relative to
// the target coordinate space, produces the strings needed to index into the
// source coordinate space if it is represented as a multidimensional array.
//
// For example,
//
//    trans has source shape (2,2,2) stride (2,2,2), axis order (2,0,1),
//       and start offsets (3,4,5)
//
//    index_vars are "i", "j", "k"
//
// this will produce:
//
//    "((4 * ((k) * 2 + 5)) + (2 * ((i) * 2 + 3)) + ((j) * 2 + 4))"
//
//
std::string ngraph::runtime::cpu::kernels::emit_linear_index(CoordinateTransform trans,
                                                             std::vector<std::string> index_vars)
{
    std::vector<std::string> multi_indices = emit_multi_indices(trans, index_vars);

    size_t stride = 1;

    for (size_t i = index_vars.size(); i-- > 0;)
    {
        // No need to do this (multiply by stride) if it's 1, though it wouldn't hurt anything.
        if (stride != 1)
        {
            std::stringstream ss;
            ss << "(" << stride << " * " << multi_indices[i] << ")";
            multi_indices[i] = ss.str();
        }

        stride *= trans.get_source_shape()[i];
    }

    std::stringstream ss;
    ss << "(" << join(multi_indices, " + ") << ")";

    return ss.str();
}

//
// Begins an indexing loop (just a for-loop) with index_var as the index
// variable, starting at start, continuing while [index_var] < [end].
//
// Optionally emits an OpenMP parallel pragma, if "omp" is true.
//
std::string ngraph::runtime::cpu::kernels::start_index_loop(std::string index_var,
                                                            size_t start,
                                                            size_t end,
                                                            bool omp)
{
    std::stringstream ss;

    if (omp)
    {
        ss << "#pragma omp parallel for\n";
    }

    ss << "for(size_t " << index_var << " = " << start << "; " << index_var << " < " << end << "; "
       << index_var << "++)\n"
       << "{\n";

    return ss.str();
}

//
// Ends an indexing loop on the index variable [index_var].
//
std::string ngraph::runtime::cpu::kernels::end_index_loop(std::string index_var)
{
    std::stringstream ss;

    ss << "} // end for(" << index_var << ")\n";

    return ss.str();
}

std::string ngraph::runtime::cpu::kernels::emit_nd_sizes(CoordinateTransform trans)
{
    std::stringstream ss;

    for (size_t s : trans.get_source_shape())
    {
        ss << "[" << s << "]";
    }

    return ss.str();
}

std::string ngraph::runtime::cpu::kernels::emit_nd_index(CoordinateTransform trans,
                                                         std::vector<std::string> index_vars)
{
    std::stringstream ss;

    for (std::string index : emit_multi_indices(trans, index_vars))
    {
        ss << "[" << index << "]";
    }

    return ss.str();
}

//
// Emits a pointwise copy from source_buffer mediated by in_trans, to
// dest_buffer mediated by dest_trans.
//
void ngraph::runtime::cpu::kernels::emit_pointwise_copy(codegen::CodeWriter& writer,
                                                        std::string element_type,
                                                        std::string source_buffer,
                                                        std::string dest_buffer,
                                                        CoordinateTransform source_trans,
                                                        CoordinateTransform dest_trans)
{
    std::vector<std::string> index_vars;

    Shape source_start_corner = source_trans.get_source_start_corner();
    Shape source_end_corner = source_trans.get_source_end_corner();

    size_t n_axes = source_start_corner.size();

    std::string source_nd_name = writer.generate_temporary_name("source_nd");
    std::string dest_nd_name = writer.generate_temporary_name("dest_nd");

    writer << element_type << "(&" << source_nd_name << ")" << emit_nd_sizes(source_trans)
           << " = *reinterpret_cast<" << element_type << "(*)" << emit_nd_sizes(source_trans)
           << ">(" << source_buffer << ");\n";
    writer << element_type << "(&" << dest_nd_name << ")" << emit_nd_sizes(dest_trans)
           << " = *reinterpret_cast<" << element_type << "(*)" << emit_nd_sizes(dest_trans) << ">("
           << dest_buffer << ");\n";

    for (size_t i = 0; i < n_axes; i++)
    {
        std::string index_var = writer.generate_temporary_name("i");

        writer << start_index_loop(index_var, source_start_corner[i], source_end_corner[i], i == 0);
        writer.indent++;

        index_vars.push_back(index_var);
    }

    writer << dest_nd_name << emit_nd_index(dest_trans, index_vars) << " = " << source_nd_name
           << emit_nd_index(source_trans, index_vars) << ";\n";

    for (size_t i = n_axes; i-- > 0;)
    {
        writer.indent--;
        writer << end_index_loop(index_vars[i]);
    }
}
