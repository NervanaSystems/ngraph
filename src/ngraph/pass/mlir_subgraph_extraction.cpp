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

#include "mlir_subgraph_extraction.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/get_output_element.hpp"

using namespace ngraph::descriptor;
using namespace ngraph::op;
using namespace ngraph::pass;

#define TI(x) std::type_index(typeid(x))

bool MLIRSubgraphExtractionPass::run_on_function(std::shared_ptr<Function> func)
{
    // Create a CompiledKernel for all the ops in the function, except Parameters and Results.
    NodeVector ck_ops;
    for (auto op : func->get_ordered_ops())
    {
        if (TI(Parameter) != TI(*op) && TI(Result) != TI(*op))
        {
            ck_ops.push_back(op);
        }
    }

    NodeVector ck_args;
    for (auto& param : func->get_parameters())
    {
        ck_args.push_back(param);
    }

    NodeVector ck_outputs = std::move(get_subgraph_outputs(ck_ops, {} /*exclusions*/));
    NGRAPH_ASSERT(ck_outputs.size() == 1) << "Unsupported subgraph with multiple outputs";

    auto ck = std::make_shared<CompiledKernel>(ck_ops, ck_outputs, ck_args);

    // Connect CompiledKernel to output nodes by replacing the output descriptors of the output
    // nodes.
    for (size_t i = 0, end = ck_outputs.size(); i < end; ++i)
    {
        auto& output_descs = ck_outputs[i]->get_outputs();
        NGRAPH_ASSERT(output_descs.size() == 1) << "Unexpected multiple output descriptors";
        auto& out_desc = output_descs[0];

        // 'replace_output' invalidates iterator of the original container. Use a copy instead.
        std::set<Input*> input_descs{out_desc.get_inputs()};

        for (Input* in_desc : input_descs)
        {
            in_desc->replace_output(ck, i);
        }
    }

    return true;
}
