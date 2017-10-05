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

#include <fstream>

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::descriptor;

pass::DumpSorted::DumpSorted(const string& output_file)
    : m_output_file{output_file}
{
}

bool pass::DumpSorted::run_on_call_list(list<Node*>& nodes)
{
    ofstream out{m_output_file};
    if (out)
    {
        for (const Node* node : nodes)
        {
            out << node->get_name() << "(";
            vector<string> inputs;
            for (const Input& input : node->get_inputs())
            {
                inputs.push_back(input.get_tensor().get_name());
            }
            out << join(inputs);
            out << ") -> ";

            vector<string> outputs;
            for (const Output& output : node->get_outputs())
            {
                outputs.push_back(output.get_tensor().get_name());
            }
            out << join(outputs);
            out << "\n";

            for (const Tensor* tensor : node->liveness_live_list)
            {
                out << "    L " << tensor->get_name() << "\n";
            }
            for (const Tensor* tensor : node->liveness_new_list)
            {
                out << "    N " << tensor->get_name() << "\n";
            }
            for (const Tensor* tensor : node->liveness_free_list)
            {
                out << "    F " << tensor->get_name() << "\n";
            }
        }
    }

    return false;
}
