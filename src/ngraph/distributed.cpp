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

template <>
std::string ngraph::as_type<std::string>(reduction::Type value)
{
    switch (value)
    {
    case reduction::Type::SUM: return "SUM";
    case reduction::Type::PROD: return "PROD";
    case reduction::Type::MIN: return "MIN";
    case reduction::Type::MAX: return "MAX";
    }
}

template <>
reduction::Type ngraph::as_type<reduction::Type>(const std::string& value)
{
    reduction::Type result = reduction::Type::SUM;
    if (value == "SUM")
    {
        result = reduction::Type::SUM;
    }
    else if (value == "PROD")
    {
        result = reduction::Type::PROD;
    }
    else if (value == "MIN")
    {
        result = reduction::Type::MIN;
    }
    else if (value == "MAX")
    {
        result = reduction::Type::MAX;
    }
    else
    {
        NGRAPH_DEBUG << "Invalid reduction type: " << value;
    }
    return result;
}

std::ostream& reduction::operator<<(std::ostream& out, const reduction::Type& obj)
{
    return out << as_type<std::string>(obj);
}

static std::unique_ptr<DistributedInterface> s_distributed_interface;

void ngraph::set_distributed_interface(std::unique_ptr<DistributedInterface> distributed_interface)
{
    NGRAPH_DEBUG << "Setting distributed interface to: " << distributed_interface->get_name();
    s_distributed_interface = std::move(distributed_interface);
}

DistributedInterface* ngraph::get_distributed_interface()
{
    if (nullptr == s_distributed_interface)
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
