/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief %Function call operation.
        ///
        /// ## Parameters
        ///
        /// |            | Description                |
        /// | ---------- | -------------------------- |
        /// | `function` | The function to be called. |
        ///
        /// ## Inputs
        ///
        /// |        | Type                                                                                                                                                                       | Description                          |
        /// | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
        /// | `args` | \f$T_1,\dots,T_n\f$ where \f$n\f$ matches the number of arguments expected by `function` and \f$T_i\f$ matches the type expected for the \f$i\f$th argument of `function`. | The arguments for the function call. |
        ///
        /// ## Output
        ///
        /// | Type      | Description                                              |
        /// | --------- | -------------------------------------------------------- |
        /// | \f$T_R\f$ | The tensor returned by `function` when called on `args`. |
        class FunctionCall : public Node
        {
        public:
            /// \brief Constructs a function call operation.
            ///
            /// \param function The function to be called.
            /// \param args The arguments for the function call.
            FunctionCall(std::shared_ptr<Function> function, const Nodes& args);

            virtual std::shared_ptr<Node> copy_with_new_args(const Nodes& new_args) const override
            {
                return std::make_shared<FunctionCall>(m_function, new_args);
            }

            /// \return A singleton vector containing the function to be called.
            std::vector<std::shared_ptr<Function>> get_functions() const override
            {
                return std::vector<std::shared_ptr<Function>>{m_function};
            }

        protected:
            std::shared_ptr<Function> m_function;
        };
    }
}
