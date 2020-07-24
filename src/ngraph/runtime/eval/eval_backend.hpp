//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/reference/allreduce.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace eval
        {
            class EVALBackend;
            class EVALExecutable;
        }
    }
}

class ngraph::runtime::eval::EVALBackend : public Backend
{
public:
    EVALBackend();
    EVALBackend(const EVALBackend&) = delete;
    EVALBackend(EVALBackend&&) = delete;
    EVALBackend& operator=(const EVALBackend&) = delete;

    std::shared_ptr<Tensor> create_tensor() override;

    std::shared_ptr<Tensor>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<Tensor> create_tensor(const element::Type& type, const Shape& shape) override;

    std::shared_ptr<Executable> compile(std::shared_ptr<Function> function,
                                        bool enable_performance_data = false) override;
};
