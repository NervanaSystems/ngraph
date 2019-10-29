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

#include "ngraph/lambda.hpp"

using namespace std;
using namespace ngraph;

constexpr DiscreteTypeInfo Lambda::type_info;

Lambda::Lambda(const OutputVector& results, const ParameterVector& parameters)
    : Lambda(as_result_vector(results), parameters)
{
}

Lambda::Lambda(const ResultVector& results, const ParameterVector& parameters)
    : m_results(results)
    , m_parameters(parameters)
{
}
