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

#ifdef NGRAPH_DISTRIBUTED

#include <mlsl.hpp>

#include "ngraph/distributed.hpp"

using namespace ngraph;

ngraph::Distributed::Distributed()
{
    if (!MLSL::Environment::GetEnv().IsInitialized())
    {
        MLSL::Environment::GetEnv().Init(nullptr, nullptr);
    }
}

ngraph::Distributed::~Distributed()
{
    if (MLSL::Environment::GetEnv().IsInitialized())
    {
        MLSL::Environment::GetEnv().Finalize();
    }
}

size_t ngraph::Distributed::get_size() const
{
    return MLSL::Environment::GetEnv().GetProcessCount();
}

size_t ngraph::Distributed::get_rank() const
{
    return MLSL::Environment::GetEnv().GetProcessIdx();
}
#endif
