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

#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace vp = vertexai::plaidml;

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // Parameter binds a descriptor::Tensor to a PlaidML Placeholder.
            template <>
            void Impl<op::Parameter>::operator()()
            {
                check_inputs(0);
                check_outputs(1);
                vp::placeholder ph{build()->io_dim_override ? build()->io_dim_override_count
                                                            : op().get_output_shape(0).size()};
                std::string name = std::string{"I"} + std::to_string(build()->input_names.size());
                descriptor::Tensor* tv = op().get_output_tensor_ptr().get();
                build()->bindings.emplace(tv, TensorInfo{ph, TensorContents::DATA});
                build()->composer.input(name, ph);
                build()->input_names.emplace(tv, std::move(name));
            }

            // Result binds a PlaidML variable to a composed function output.
            template <>
            void Impl<op::Result>::operator()()
            {
                check_inputs(1);
                check_outputs(1);
                std::string name = std::string{"O"} + std::to_string(build()->output_names.size());
                descriptor::Tensor* tv = op().get_output_tensor_ptr().get();
                build()->composer.output(name, op_input());
                build()->output_names.emplace(tv, std::move(name));
            }

            namespace
            {
                Impl<op::Parameter>::Registration register_parameter;
                Impl<op::Result>::Registration register_result;
            }
        }
    }
}
