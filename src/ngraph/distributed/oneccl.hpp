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

#include "ngraph/distributed.hpp"

#ifdef NGRAPH_DISTRIBUTED_ONECCL_ENABLE
#include <string>
#include <ccl.hpp>

namespace ngraph
{
    namespace distributed
    {
        class CCLDistributedInterface : public DistributedInterface
        {
        public:
            CCLDistributedInterface(const std::string& name = "OneCCL")
                : m_name(name)
            {
                m_stream = ccl::environment::instance().create_stream();
                m_comm = ccl::environment::instance().create_communicator();
            }

            ~CCLDistributedInterface() override = default;

            const std::string& get_name() const override { return m_name; }
            int get_size() override
            {
                return static_cast<int>(m_comm->size());
            }

            int get_rank() override
            {
                return static_cast<int>(m_comm->rank());
            }

            void log_print(const std::string& timestamp, const std::vector<char>& buf) override
            {
                std::printf("%s [CCL RANK: %d]: %s\n", timestamp.c_str(), get_rank(), buf.data());
            }

            void all_reduce(void* in,
                            void* out,
                            element::Type_t element_type,
                            reduction::Type reduce_type,
                            size_t count) override
            {
                auto data_type = toCCLDataType(element_type);
                auto reduction = toCCLReduceType(reduce_type);
                m_comm->allreduce(in, out, count, data_type, reduction, nullptr, m_stream)->wait();
            }

            void broadcast(void* in,
                           element::Type_t element_type,
                           size_t count,
                           int root_id) override
            {
                auto data_type = toCCLDataType(element_type);
                m_comm->bcast(in, count, data_type, static_cast<size_t>(root_id), nullptr, m_stream)->wait();
            }

            void recv(void* /* in */,
                      element::Type_t /* element_type */,
                      size_t /* count */,
                      int /* src_id */) override
            {
                throw ngraph_error("recv not supported in CCL");
            }

            void send(const void* /* in */,
                      element::Type_t /* element_type */,
                      size_t /* count */,
                      int /* dest_id */) override
            {
                throw ngraph_error("send not supported in CCL");
            }

        private:
            ccl::data_type toCCLDataType(element::Type_t element_type)
            {
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
                switch (element_type)
                {
                case element::Type_t::bf16: return ccl::data_type::dt_bfp16;
                case element::Type_t::f32: return ccl::data_type::dt_float;
                case element::Type_t::f64: return ccl::data_type::dt_double;
                case element::Type_t::i64: return ccl::data_type::dt_int64;
                case element::Type_t::u64: return ccl::data_type::dt_uint64;
                default:
                    throw std::runtime_error(
                        "data type not supported in CCL");
                }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
            }

            ccl::reduction toCCLReduceType(reduction::Type reduce_type)
            {
                #if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
                switch (reduce_type)
                {
                case reduction::Type::SUM: return ccl::reduction::sum; break;
                case reduction::Type::PROD: return ccl::reduction::prod; break;
                case reduction::Type::MIN: return ccl::reduction::min; break;
                case reduction::Type::MAX: return ccl::reduction::max; break;
                }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

            }

        protected:
            std::string m_name{"OneCCL"};
            ccl::stream_t m_stream;
            ccl::communicator_t m_comm;
        };
    }
}
#endif
