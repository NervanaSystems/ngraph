#pragma once
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

#include <string>
#include <vector>

class NGraph
{
public:
    void add_params(const std::vector<std::string>& paramList);
    const std::vector<std::string>& get_params() const;
    std::string                     get_name() const { return "NGraph Implementation Object"; }
private:
    std::vector<std::string> m_params;
};

// Factory methods
extern "C" NGraph* create_ngraph_object();
extern "C" void destroy_ngraph_object(NGraph* pObj);

// FUnction pointers to the factory methods
typedef NGraph* (*CreateNGraphObjPfn)();
typedef void (*DestroyNGraphObjPfn)(NGraph*);
