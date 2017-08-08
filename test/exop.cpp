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

#include <vector>
#include <string>
#include <sstream>

#include "gtest/gtest.h"

#include "transformers/exop.hpp"
#include "transformers/mock.hpp"
#include "transformers/mock_transformer.hpp"

using namespace ngraph;

TEST(exop, create)
{
    // CpuTransformer      transformer;

    // op_ptr c1 = constant(1.0);
    // op_ptr a1 = add(c1, c1);

    // std::vector<op_ptr> inputs;
    // std::vector<op_ptr> outputs = {a1};

    // auto computation_op = std::make_shared<ComputationOp>(inputs, outputs);

    // ExecutionState&     es = transformer.execution_state();
    // execution_graph_ptr eg = es.make_execution_graph(computation_op); // one at a time
    // computation_decl_ptr    cd = eg->computation_decl;
    // // transforer.run_passes(computation_decl_ptr);
}
