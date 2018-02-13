// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
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

#include <memory>
#include <vector>

#include <mkldnn.hpp>

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;

            class MKLDNNEmitter
            {
            public:
                MKLDNNEmitter(std::shared_ptr<CPU_ExternalFunction> ef)
                    : external_function(ef)
                {
                }

                void build_memory_descriptor();

            private:
                std::shared_ptr<CPU_ExternalFunction> external_function;
                std::vector<mkldnn::primitive> mkldnn_primitives;
            };
        }
    }
}
