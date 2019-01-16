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

namespace ngraph
{
    class Distributed
    {
    public:
        Distributed() {};
        virtual ~Distributed() = 0;
        virtual void is_initialized() = 0;
        virtual size_t get_process_count() = 0 const;
        virtual size_t get_process_id() = 0 const;
        // return data_type
        virtual auto data_type(std::string type) = 0 const;

    };

    // implementation with MPI
    class Distributed_MPI : public Distributed
    {
    public:
        Distributed();
        ~Distributed();
        size_t get_process_count() const;
        size_t get_process_id() const;

    };

    // implementation with MLSL
    // split the classes in separate files and put in the right place
    // if distributed_mlsl.hpp and .cpp belongs to cpu runtime
    class Distributed_MLSL : public Distributed
    {
    public:
        Distributed();
        ~Distributed();
        size_t get_process_count() const;
        size_t get_process_id() const;
    };
    
}
