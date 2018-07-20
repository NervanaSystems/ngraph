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

#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Abstract base class for elementwise unary operations, i.e., operations where the same
            ///        scalar operation is applied to each element.
            ///
            /// For example, if the underlying operation (determined by the subclass) is \f$\mathit{op}(x)\f$, the input tensor
            /// \f$[[x,y],[z,w]]\f$ will be mapped to \f$[[\mathit{op}(x),\mathit{op}(y)],[\mathit{op}(z),\mathit{op}(w)]]\f$.
            ///
            /// ## Inputs
            ///
            /// |       | Type                              | Description                                                                                                                                   |
            /// | ----- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
            /// | `arg` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape. Subclasses may impose restrictions on the element type \f$E\f$. |
            ///
            /// ## Output
            ///
            /// | Type                    | Description                                                                                                                                                                                                                                                            |
            /// | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
            /// | \f$E'[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg}[i_1,\dots,i_n])\f$. This will always have the same shape as the input tensor, but subclasses must determine the element type \f$E'\f$. |
            class UnaryElementwise : public RequiresTensorViewArgs
            {
            protected:
                /// \brief Constructs a unary elementwise tensor operation.
                ///
                /// \param arg Node that produces the input tensor.
                UnaryElementwise(const std::string& node_type, const std::shared_ptr<Node>& arg);

                void validate_and_infer_types() override;
            };
        }
    }
}
