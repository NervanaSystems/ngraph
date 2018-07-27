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

#include <CPP/batch_norm.hpp>
#include <CPP/concatenation.hpp>
#include <CPP/scale.hpp>
#include <CPP/split.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_batchnorm.hpp"

#include "ngraph/op/batch_norm.hpp"

using namespace std;
using namespace ngraph;

static string do_matrix_split(cldnn::topology& topology,
                              const string& name,
                              const vector<pair<cldnn::primitive_id, cldnn::tensor>>& offsets)
{
    const string result = name + "_split";

    const cldnn::split op_split(result, name, offsets);
    topology.add(op_split);
    return result;
}

static string get_batch_norm_mean(cldnn::topology& topology, const string& input_name)
{
    throw invalid_argument(
        "intelgpu::get_batch_norm_mean() Calculation matrix mean is not yet supported.");
}

static string get_batch_norm_variance(cldnn::topology& topology,
                                      const string& input_name,
                                      const string& mean_name)
{
    throw invalid_argument(
        "intelgpu::get_batch_norm_variance() Calculation matrix variance is not yet supported.");
}

void runtime::intelgpu::do_batch_norm_operation(cldnn::topology& topology,
                                                const string& output_name,
                                                double eps,
                                                const string& input_name,
                                                const Shape& input_shape,
                                                const string& gamma_name,
                                                const string& beta_name,
                                                const string& mean_name_inp,
                                                const string& variance_name_inp)
{
    vector<pair<cldnn::primitive_id, cldnn::tensor>> split_offsets;
    vector<pair<cldnn::primitive_id, cldnn::tensor>> vec_offsets;
    vector<cldnn::primitive_id> dim_set;

    if (input_shape.size() < 2 || input_shape.size() > 4)
    {
        throw invalid_argument("intelgpu::do_batch_norm_operation() wrong input shape.");
    }

    // According to the documentation, input data channel is always being axis 1
    // Assumed the second dimension from the left. Example {0, 1, 0, 0} or {0, 1}
    // Also, input data must be at least 2D array
    const size_t shape_channel = 1;
    const size_t cldnn_channel = 4 - input_shape.size() + shape_channel;
    const cldnn::concatenation::concatenation_axis direction =
        runtime::intelgpu::IntelGPULayout::get_cldnn_axis(cldnn_channel);

    const size_t split_arr_count = input_shape.at(shape_channel);
    for (size_t i = 0; i < split_arr_count; ++i)
    {
        const string str_i = to_string(i);
        const cldnn::tensor vec_offset(0, 0, i, 0);
        vec_offsets.push_back(pair<cldnn::primitive_id, cldnn::tensor>(str_i, vec_offset));

        vector<cldnn::tensor::value_type> offset({0, 0, 0, 0}); // No action by default
        offset.at(cldnn_channel) = i;

        const cldnn::tensor input_offset(offset.at(0), offset.at(1), offset.at(3), offset.at(2));
        split_offsets.push_back(pair<cldnn::primitive_id, cldnn::tensor>(str_i, input_offset));
    }

    string mean_name = mean_name_inp;
    if (mean_name_inp.empty())
    {
        mean_name = get_batch_norm_mean(topology, input_name);
    }

    string variance_name = variance_name_inp;
    if (variance_name_inp.empty())
    {
        variance_name = get_batch_norm_variance(topology, input_name, mean_name);
    }

    const string input_split_name = do_matrix_split(topology, input_name, split_offsets);
    const string mean_split_name = do_matrix_split(topology, mean_name, vec_offsets);
    const string variance_split_name = do_matrix_split(topology, variance_name, vec_offsets);
    const string gamma_split_name = do_matrix_split(topology, gamma_name, vec_offsets);
    const string beta_split_name = do_matrix_split(topology, beta_name, vec_offsets);

    for (size_t i = 0; i < split_arr_count; ++i)
    {
        const string suf = ':' + to_string(i);
        const string out_bn_name = output_name + "_out_bn";

        const cldnn::batch_norm cldd_batchnorm(out_bn_name + suf,
                                               input_split_name + suf,
                                               mean_split_name + suf,
                                               variance_split_name + suf,
                                               eps);
        topology.add(cldd_batchnorm);

        const cldnn::scale op_scale(
            output_name + suf, out_bn_name + suf, gamma_split_name + suf, beta_split_name + suf);
        topology.add(op_scale);

        dim_set.push_back(output_name + suf);
    }

    const cldnn::concatenation op_concat(output_name, dim_set, direction);
    topology.add(op_concat);
}
