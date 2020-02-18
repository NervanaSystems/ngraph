//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <memory>
#include <string>
#include <vector>

#include <CPP/topology.hpp>

#include "ngraph/node.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/gemm.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            class CustomKernelInfo;
            class CustomKernels;

            void arguments_check(const std::shared_ptr<Node>& op, size_t input, size_t output);
        }
    }
}

class ngraph::runtime::intelgpu::CustomKernelInfo
{
public:
    CustomKernelInfo(const std::string& name,
                     const Shape& shape,
                     const element::Type& type,
                     const std::vector<std::string>& inputs,
                     const std::string& code,
                     const std::string& entry_point,
                     const std::vector<size_t>& gws = {1},
                     const std::vector<size_t>& lws = {1})
    {
        m_name = name;
        m_shape = shape;
        m_type = type;
        m_inputs = inputs;
        m_code = code;
        m_entry_point = entry_point;
        m_gws = gws;
        m_lws = lws;
        kernel = nullptr;
    }

    std::string m_name;
    Shape m_shape;
    element::Type m_type;
    std::vector<std::string> m_inputs;
    std::string m_code;
    std::string m_entry_point;
    std::vector<size_t> m_gws;
    std::vector<size_t> m_lws;
    void* kernel;
};

class ngraph::runtime::intelgpu::CustomKernels
{
public:
    using krnl_info = std::vector<CustomKernelInfo>;

    explicit CustomKernels(cldnn::topology& backend_stream)
        : stream(backend_stream)
    {
        m_count_krnls = 0;
    }

    template <typename OP>
    void emit(const std::shared_ptr<OP>& op)
    {
        krnl_info krnl_info;

        krnl_info = build_krnl(op);

        queue_krnl(krnl_info, op);
    }

    size_t get_custom_kernel_count() const { return m_count_krnls; }
private:
    void queue_krnl(const krnl_info& krn_info, const std::shared_ptr<Node>& op);

    krnl_info build_krnl(const std::shared_ptr<op::All>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::And>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Any>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::BatchNormInference>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::BatchNormTraining>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::BatchNormTrainingBackprop>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Broadcast>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Convolution>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::GroupConvolution>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::ConvolutionBackpropData>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::ConvolutionBackpropFilters>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::ConvolutionBias>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::ConvolutionBiasAdd>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::ConvolutionBiasBackpropFiltersBias>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Equal>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Gemm>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Greater>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::GreaterEq>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Less>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::LessEq>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Max>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Min>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::NotEqual>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Or>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Product>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Select>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Slice>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Softmax>& op) const;
    krnl_info build_krnl(const std::shared_ptr<op::Sum>& op) const;

    cldnn::topology& stream;
    size_t m_count_krnls;
};
