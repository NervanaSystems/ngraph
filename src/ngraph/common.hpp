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

#pragma once

#include <memory>
#include <vector>
#include <set>

// Names for types that aren't worth giving their own classes
namespace ngraph
{
    class Node;
    class Parameter;

    /// Zero or more nodes
    using Nodes = std::vector<std::shared_ptr<Node>>;
    
    /// A set of indices, for example, reduction axes
    using IndexSet = std::set<size_t>;

    /// A list of parameters
    using Parameters = std::vector<std::shared_ptr<Parameter>>;
}
