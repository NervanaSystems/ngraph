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

#include "ngraph/log.hpp"
#include "ngraph/distributed/distributed.hpp"

using namespace ngraph;

ngraph::distributed::Distributed::Distributed()
{
    NGRAPH_INFO << "DistributedMLSL::Distributed() -- begin ";
    NGRAPH_INFO << "DistributedMLSL::Distributed() -- end ";
}

void ngraph::distributed::Distributed::finalize()
{
    NGRAPH_INFO << "DistributedMLSL::finalize() -- begin ";
    NGRAPH_INFO << "DistributedMLSL::finalize() -- end ";   
}

int ngraph::distributed::Distributed::get_size() const
{   
    NGRAPH_INFO << "Distributed::get_size() -- begin & end";
    return 1;
}

int ngraph::distributed::Distributed::get_rank() const
{   
    NGRAPH_INFO << "Distributed::get_rank() -- begin & end";
    return 1;
}
#endif
