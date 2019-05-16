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

#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
#include <string>

#include <mlsl.hpp>

#include "ngraph/distributed.hpp"

namespace ngraph
{
    namespace distributed
    {
        class MLSLDistributedInterface : public DistributedInterface
        {
        public:
            MLSLDistributedInterface(const std::string& name = "MLSL")
                : m_name(name)
            {
                if (!MLSL::Environment::GetEnv().IsInitialized() && !m_initialized_mlsl)
                {
                    MLSL::Environment::GetEnv().Init(nullptr, nullptr);
                    m_initialized_mlsl = true;
                }
            }

            ~MLSLDistributedInterface() override
            {
                if (MLSL::Environment::GetEnv().IsInitialized() && m_initialized_mlsl)
                {
                    MLSL::Environment::GetEnv().Finalize();
                    m_initialized_mlsl = false;
                }
            }

            const std::string& get_name() const override { return m_name; }
            int get_size() override
            {
                return static_cast<int>(MLSL::Environment::GetEnv().GetProcessCount());
            }

            int get_rank() override
            {
                return static_cast<int>(MLSL::Environment::GetEnv().GetProcessIdx());
            }

            void
                all_reduce(void* in, void* out, element::Type_t element_type, size_t count) override
            {
                auto data_type = MLSL::DT_FLOAT;

                if (element_type == element::Type_t::f32)
                {
                    data_type = MLSL::DT_FLOAT;
                }
                else if (element_type == element::Type_t::f64)
                {
                    data_type = MLSL::DT_DOUBLE;
                }
                else
                {
                    throw std::runtime_error("AllReduce op supports only f32 and f64 types");
                }

                MLSL::Environment& env = MLSL::Environment::GetEnv();
                MLSL::Distribution* distribution = env.CreateDistribution(env.GetProcessCount(), 1);
                MLSL::CommReq* req =
                    distribution->AllReduce(in, out, count, data_type, MLSL::RT_SUM, MLSL::GT_DATA);
                env.Wait(req);
                env.DeleteDistribution(distribution);
            }

            void broadcast(void* in, element::Type_t element_type, size_t count) override
            {
                auto data_type = MLSL::DT_FLOAT;

                if (element_type == element::Type_t::f64)
                {
                    data_type = MLSL::DT_DOUBLE;
                }
                else if (element_type != element::Type_t::f32)
                {
                    throw std::runtime_error(
                        "BroadcastDistributed op supports only f32 and f64 types");
                }

                MLSL::Environment& env = MLSL::Environment::GetEnv();
                MLSL::Distribution* distribution = env.CreateDistribution(env.GetProcessCount(), 1);
                MLSL::CommReq* req = distribution->Bcast(in, count, data_type, 0, MLSL::GT_DATA);
                env.Wait(req);
                env.DeleteDistribution(distribution);
            }

        protected:
            std::string m_name{"MLSL"};
            bool m_initialized_mlsl = false;
        };
    }
}
#endif
