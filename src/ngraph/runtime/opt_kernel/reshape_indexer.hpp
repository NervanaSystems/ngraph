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

#include "ngraph/axis_vector.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace opt_kernel
        {
            class ReshapeIndexer
            {
            public:
                class Indexer
                {
                public:
                    virtual ~Indexer() {}
                    virtual size_t next() = 0;
                };

                ReshapeIndexer(const Shape& in_shape, const AxisVector& in_axis_order);
                size_t next();

            private:
                std::unique_ptr<Indexer> m_indexer;
            };
        }
    }
}
