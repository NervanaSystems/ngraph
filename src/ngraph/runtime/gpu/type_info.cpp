//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/runtime/gpu/type_info.hpp"

using namespace ngraph;

const runtime::gpu::TypeInfo::TypeDispatch runtime::gpu::TypeInfo::dispatcher{
    {"char", std::make_shared<runtime::gpu::TypeInfo_Impl<char>>()},
    {"float", std::make_shared<runtime::gpu::TypeInfo_Impl<float>>()},
    {"double", std::make_shared<runtime::gpu::TypeInfo_Impl<double>>()},
    {"int8_t", std::make_shared<runtime::gpu::TypeInfo_Impl<int8_t>>()},
    {"int16_t", std::make_shared<runtime::gpu::TypeInfo_Impl<int16_t>>()},
    {"int32_t", std::make_shared<runtime::gpu::TypeInfo_Impl<int32_t>>()},
    {"int64_t", std::make_shared<runtime::gpu::TypeInfo_Impl<int64_t>>()},
    {"uint8_t", std::make_shared<runtime::gpu::TypeInfo_Impl<uint8_t>>()},
    {"uint16_t", std::make_shared<runtime::gpu::TypeInfo_Impl<uint16_t>>()},
    {"uint32_t", std::make_shared<runtime::gpu::TypeInfo_Impl<uint32_t>>()},
    {"uint64_t", std::make_shared<runtime::gpu::TypeInfo_Impl<uint64_t>>()}};
