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

#ifdef NGRAPH_DISTRIBUTED

#include <mlsl.hpp>

#include "ngraph/log.hpp"
#include "ngraph/distributed/distributed_mlsl.hpp"

using namespace ngraph;

ngraph::distributed::DistributedMLSL::DistributedMLSL()
{
    NGRAPH_INFO << "DistributedMLSL::DistributedMLSL() -- begin "; 
    if (!MLSL::Environment::GetEnv().IsInitialized())
    {
        MLSL::Environment::GetEnv().Init(nullptr, nullptr);
    }
    NGRAPH_INFO << "DistributedMLSL::DistributedMLSL() -- end "; 
}

void ngraph::distributed::DistributedMLSL::finalize()
{   
    NGRAPH_INFO << "DistributedMLSL::finalize() -- begin "; 
    if (MLSL::Environment::GetEnv().IsInitialized())
    {
        MLSL::Environment::GetEnv().Finalize();
    }
    NGRAPH_INFO << "DistributedMLSL::finalize() -- end "; 
}

int ngraph::distributed::DistributedMLSL::get_size() const
{   
    NGRAPH_INFO << "DistributedMLSL::get_size() -- begin & end"; 
    return static_cast<int>(MLSL::Environment::GetEnv().GetProcessCount());
}

int ngraph::distributed::DistributedMLSL::get_rank() const
{
    NGRAPH_INFO << "DistributedMLSL::get_rank() -- begin & end "; 
    return static_cast<int>(MLSL::Environment::GetEnv().GetProcessIdx());
}
#endif
