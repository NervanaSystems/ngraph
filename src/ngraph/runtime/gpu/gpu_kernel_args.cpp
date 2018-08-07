/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <string>

#include "ngraph/runtime/gpu/gpu_kernel_args.hpp"

#define TI(x) std::type_index(typeid(x))

using namespace ngraph;

const std::unordered_map<std::type_index, std::string> runtime::gpu::GPUKernelArgs::type_names = {
    {TI(size_t), "size_t"},
    {TI(char), "char"},
    {TI(float), "float"},
    {TI(double), "double"},
    {TI(int8_t), "int8_t"},
    {TI(int16_t), "int16_t"},
    {TI(int32_t), "int32_t"},
    {TI(int64_t), "int64_t"},
    {TI(uint8_t), "uint8_t"},
    {TI(uint16_t), "uint16_t"},
    {TI(uint32_t), "uint32_t"},
    {TI(uint64_t), "uint64_t"}};

runtime::gpu::GPUKernelArgs::GPUKernelArgs(const std::shared_ptr<GPUHostParameters>& params)
    : m_signature_generated(false)
    , m_host_parameters(params)
{
    m_input_signature << "(";
}

runtime::gpu::GPUKernelArgs::GPUKernelArgs(const GPUKernelArgs& args)
{
    m_signature_generated = args.m_signature_generated;
    m_argument_list = args.m_argument_list;
    m_placeholder_positions = args.m_placeholder_positions;
    m_input_signature << args.m_input_signature.str();
    m_host_parameters = args.m_host_parameters;
}

void runtime::gpu::GPUKernelArgs::validate()
{
    if (m_signature_generated)
    {
        throw std::runtime_error(
            "Kernel input signature already generated. Adding additional kernel arguments has no "
            "effect.");
    }
}

void runtime::gpu::GPUKernelArgs::add_to_signature(const std::string& type, const std::string& name)
{
    if (m_input_signature.str() == "(")
    {
        m_input_signature << type << " " << name;
    }
    else
    {
        m_input_signature << ", " << type << " " << name;
    }
}

runtime::gpu::GPUKernelArgs& runtime::gpu::GPUKernelArgs::add_placeholder(const std::string& type,
                                                                          const std::string& name)
{
    validate();
    m_argument_list.push_back(nullptr);
    m_placeholder_positions.push_back(true);
    add_to_signature(type + "*", name);
    return *this;
}

runtime::gpu::GPUKernelArgs& runtime::gpu::GPUKernelArgs::resolve_placeholder(size_t arg_num,
                                                                              void* address)
{
    if (m_placeholder_positions.at(arg_num))
    {
        m_argument_list[arg_num] = address;
    }
    else
    {
        throw std::runtime_error("Resolution of specified non-placeholder argument is unallowed.");
    }
    return *this;
}

std::string runtime::gpu::GPUKernelArgs::get_input_signature()
{
    if (m_signature_generated == false)
    {
        m_signature_generated = true;
        m_input_signature << ")";
    }
    return m_input_signature.str();
}
