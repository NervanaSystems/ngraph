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

#include <CPP/concatenation.hpp>
#include <CPP/layout.hpp>
#include <CPP/tensor.hpp>

#include "ngraph/descriptor/layout/tensor_view_layout.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            class IntelGPULayout;
        }
    }
}

class ngraph::runtime::intelgpu::IntelGPULayout
    : public ngraph::descriptor::layout::TensorViewLayout
{
public:
    IntelGPULayout(const ngraph::descriptor::TensorView& tv, const cldnn::layout& layout);
    ~IntelGPULayout() override {}
    size_t get_size() override { return cldnn_layout.get_linear_size(); }
    size_t get_index_offset(const std::vector<size_t>& indices) override;

    const Strides& get_strides() const override { return strides; }
    bool operator==(const TensorViewLayout& other) const override;

    void set_cldnn_layout(const cldnn::layout& layout) { cldnn_layout = layout; }
    cldnn::layout get_cldnn_layout() const { return cldnn_layout; }
    static cldnn::data_types get_cldnn_type(const ngraph::element::Type& element_type);
    static cldnn::layout create_cldnn_layout(const ngraph::element::Type& element_type,
                                             const Shape& element_shape);
    static cldnn::tensor create_cldnn_tensor(const Shape& element_shape);
    static cldnn::tensor create_cldnn_offset(const Shape& pad_below);
    // This function converts Shape dimension_id into cldnn::concatenation id
    static cldnn::concatenation::concatenation_axis get_cldnn_axis(size_t shape_size, size_t axis);

private:
    Strides strides;
    cldnn::layout cldnn_layout;
};
