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

#include <cstddef>
#include <memory>
#include <string>

#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    class DistributedInterface
    {
    public:
        virtual ~DistributedInterface() {}
        virtual const std::string& get_name() const = 0;
        virtual int get_size() = 0;
        virtual int get_rank() = 0;
        virtual void log_print(const std::string& timestamp, const std::vector<char>& buf) = 0;

        virtual void
            all_reduce(void* in, void* out, element::Type_t element_type, size_t count) = 0;
        virtual void broadcast(void* in, element::Type_t element_type, size_t count) = 0;
    };

    void set_distributed_interface(std::unique_ptr<DistributedInterface> distributed_interface);
    DistributedInterface* get_distributed_interface();
}
