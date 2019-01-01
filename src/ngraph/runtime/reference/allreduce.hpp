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

#pragma once

#ifdef NGRAPH_DISTRIBUTED

#include <mlsl.hpp>

#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void allreduce(T* arg, T* out, const element::Type element_type, int count)
            {
                auto data_type = MLSL::DT_FLOAT;

                if (element_type == element::f32)
                {
                    data_type = MLSL::DT_FLOAT;
                }
                else if (element_type == element::f64)
                {
                    data_type = MLSL::DT_DOUBLE;
                }
                else
                {
                    throw std::runtime_error("AllReduce op supports only f32 and f64 types");
                }

                MLSL::Environment& env = MLSL::Environment::GetEnv();
                MLSL::Distribution* distribution = env.CreateDistribution(env.GetProcessCount(), 1);
                MLSL::CommReq* req = distribution->AllReduce(
                    arg, out, count, data_type, MLSL::RT_SUM, MLSL::GT_DATA);
                env.Wait(req);
                env.DeleteDistribution(distribution);
            }
        }
    }
}

#endif
