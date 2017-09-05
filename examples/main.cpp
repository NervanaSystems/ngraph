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

#include <stdio.h>
#include "ngraph/ngraph.hpp"
#include "ngraph/ops/dot.hpp"

using namespace std;
using namespace ngraph;

int main(int argc, char** argv)
{
    printf( "Building graph\n" );

    // Function with 4 parameters
    auto arg0        = op::parameter(element::Float::type, {7, 3});
    auto arg1        = op::parameter(element::Float::type, {3});
    auto arg2        = op::parameter(element::Float::type, {32, 7});
    auto arg3        = op::parameter(element::Float::type, {32, 7});
    auto broadcast_1 = op::broadcast(arg3, {10, 32, 7}, {0});
    auto dot         = op::dot(arg2, arg0);
    
    auto cluster_0 = op::function(dot, {arg0, arg1, arg2, arg3});
    auto result = cluster_0->result();

    printf( "Finished\n" );
    
}