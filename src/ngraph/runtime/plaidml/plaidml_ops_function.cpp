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

#include "ngraph/op/function_call.hpp"
#include "ngraph/runtime/plaidml/plaidml_build.hpp"
#include "ngraph/runtime/plaidml/plaidml_compiler.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // FunctionCall invokes a sub-function.
            template <>
            void Impl<op::FunctionCall>::operator()()
            {
                Build b;
                build()->compiler->build(op().get_functions()[0], &b);
                vertexai::plaidml::function f{b.composer};
                vertexai::plaidml::function::parameters_t inputs;
                for (std::size_t idx = 0; idx < op().get_input_size(); ++idx)
                {
                    auto* oitv = op().get_inputs()[idx].get_output().get_tensor_ptr().get();
                    auto* iitv =
                        b.func->get_parameters()[idx]->get_outputs()[0].get_tensor_ptr().get();
                    inputs.emplace_back(b.input_names.at(iitv), build()->bindings.at(oitv).var);
                }
                vertexai::plaidml::application app{f.apply(inputs)};
                for (std::size_t idx = 0; idx < op().get_output_size(); ++idx)
                {
                    auto* iotv = b.func->get_results()[idx]->get_output_tensor_ptr().get();
                    set_output(idx, app.get_output(b.output_names[iotv]));
                }
            }

            namespace
            {
                Impl<op::FunctionCall>::Registration register_function_call;
            }
        }
    }
}
