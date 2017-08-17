// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
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
#include "ngraph.hpp"
#include "log.hpp"

NGraph* create_plugin()
{
    return new NGraph();
}

void destroy_plugin(NGraph* pObj)
{
    delete pObj;
}

void NGraph::add_params( const std::vector<std::string>& paramList )
{
    INFO << "Adding parameters";
    m_params.insert(m_params.end(), paramList.begin(), paramList.end());    
}

const std::vector<std::string>& NGraph::get_params() const
{
    return m_params;
}
