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

#include <sstream>

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/propagate_types.hpp"

using namespace std;
using namespace ngraph;

bool pass::PropagateTypes::run_on_call_graph(list<Node*>& nodes)
{
    for (Node* node : nodes)
    {
        try
        {
            NGRAPH_INFO;
            node->propagate_types();
            NGRAPH_INFO;
        }
        catch (exception& e)
        {
            stringstream ss;
            ss << "Error with node " << *node << ": ";
            ss << e.what();
            throw invalid_argument(ss.str());
        }
    }
    return false;
}
