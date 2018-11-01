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

#include "ngraph/op/and.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // And performs a simple elementwise logical and.
            template <>
            void Impl<op::And>::operator()()
            {
                check_inputs(2);
                check_outputs(1);
                set_output(start_tile_function()
                               .add(builder::Input{op_input(0, TensorContents::LOGICAL), "A"})
                               .add(builder::Input{op_input(1, TensorContents::LOGICAL), "B"})
                               .add(builder::Output{"C"})
                               .add(builder::Elementwise{"C", "A ? B : A"})
                               .finalize(),
                           TensorContents::LOGICAL);
            }

            // Not performs a simple elementwise logical not.
            template <>
            void Impl<op::Not>::operator()()
            {
                check_inputs(1);
                check_outputs(1);
                set_output(start_tile_function()
                               .add(builder::Input{op_input(0, TensorContents::LOGICAL), "I"})
                               .add(builder::Output{"O"})
                               .add(builder::Elementwise{"O", "cmp_eq(I, 0)"})
                               .finalize(),
                           TensorContents::LOGICAL);
            }

            // Or performs a simple elementwise logical or.
            template <>
            void Impl<op::Or>::operator()()
            {
                check_inputs(2);
                check_outputs(1);
                set_output(start_tile_function()
                               .add(builder::Input{op_input(0, TensorContents::LOGICAL), "A"})
                               .add(builder::Input{op_input(1, TensorContents::LOGICAL), "B"})
                               .add(builder::Output{"C"})
                               .add(builder::Elementwise{"C", "A ? A : B"})
                               .finalize(),
                           TensorContents::LOGICAL);
            }

            namespace
            {
                Impl<op::And>::Registration register_and;
                Impl<op::Not>::Registration register_not;
                Impl<op::Or>::Registration register_or;
            }
        }
    }
}
