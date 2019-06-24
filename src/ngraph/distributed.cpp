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

#include "ngraph/distributed.hpp"
#include "ngraph/distributed/mlsl.hpp"
#include "ngraph/distributed/null.hpp"
#include "ngraph/distributed/open_mpi.hpp"
#include "ngraph/log.hpp"

using namespace ngraph;

std::ostream& reduction::operator<<(std::ostream& out, const reduction::Type& obj)
{
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (obj)
    {
    case reduction::Type::SUM: out << "SUM"; break;
    case reduction::Type::PROD: out << "PROD"; break;
    case reduction::Type::MIN: out << "MIN"; break;
    case reduction::Type::MAX: out << "MAX"; break;
    }
#if !(defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
    return out;
};

static std::unique_ptr<DistributedInterface> s_distributed_interface;

void ngraph::set_distributed_interface(std::unique_ptr<DistributedInterface> distributed_interface)
{
    NGRAPH_DEBUG << "Setting distributed interfsce to: " << distributed_interface->get_name();
    s_distributed_interface = std::move(distributed_interface);
}

DistributedInterface* ngraph::get_distributed_interface()
{
    if (0 == s_distributed_interface)
    {
#ifdef NGRAPH_DISTRIBUTED_OMPI_ENABLE
        set_distributed_interface(std::unique_ptr<DistributedInterface>(
            new ngraph::distributed::OpenMPIDistributedInterface()));
#elif defined(NGRAPH_DISTRIBUTED_MLSL_ENABLE)
        set_distributed_interface(std::unique_ptr<DistributedInterface>(
            new ngraph::distributed::MLSLDistributedInterface()));
#else
        set_distributed_interface(std::unique_ptr<DistributedInterface>(
            new ngraph::distributed::NullDistributedInterface()));
#endif
    }
    return s_distributed_interface.get();
}
