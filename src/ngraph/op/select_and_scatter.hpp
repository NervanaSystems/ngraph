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

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Select-and-scatter operation.
        ///
        /// Select-and-scatter takes three inputs, all of which must have the same element type \f$E\f$:
        ///
        /// 1. the <i>selectee</i>, a tensor of shape \f$(d_1,\dots,d_n)\f$,
        /// 2. the <i>source</i>, a tensor of shape \f$(d'_1,\dots,d'_n)\f$ where \f$d'_i = \lceil \frac {d_i - w_i + 1}{s_i} \rceil\f$ (see below for definition of window sizes \f$w_i\f$ and strides \f$s_i\f$, and
        /// 3. the <i>initial value</i>, a scalar.
        ///
        /// It also takes four parameters:
        ///
        /// 1. the <i>selection function</i>, a function that takes two arguments of type \f$E[]\f$ and returns `Bool` (think of this as a binary relation),
        /// 2. the <i>scatter function</i>, a function that takes two arguments of type \f$E[]\f$ and returns \f$E[]\f$,
        /// 3. the <i>window shape</i>, a vector \f$(w_1,\dots,w_n)\f$ of non-negative integers, and
        /// 4. the <i>window movement strides</i>, a vector \f$(s_1,\dots,s_n)\f$ of non-negative integers.
        ///
        /// It is assumed that the selection function is a strict total order; otherwise behavior is undefined. (TODO: we may be able to generalize usefully here.)
        ///
        /// The output \f$T_\textit{out}\f$ has the same element type and shape as the selectee. Its values are produced as follows:
        /// 1. Initialize every element \f$T_\textit{out}\f$ with the initial value.
        /// 2. Slide a window of shape \f$(w_1,\dots,w_n)\f$ over the selectee, with stride \f$(s_1,\dots,s_n)\f$, with the start corner of the window increasing in natural ("row-major") order. Note that for every valid window position, there will be a corresponding value in the source tensor located at some coordinate \f$(i_1,\dots,i_n)\f$.
        /// 3. At each window position, using the selection function as the relation, find a coordinate \f$(j_1,\dots,j_n)\f$ where some "maximum" value resides. Replace \f$T_\textit{out}[j_1,\dots,j_n]\f$ with the value \f$f(T_\textit{out}[j_1,\dots,j_n],T_\textit{source}[i_1,\dots,i_n])\f$ where \f$f\f$ is the scatter function and \f$T_\textit{source}\f$ is the source tensor.
        ///
        /// The XLA documentation has good examples at https://www.tensorflow.org/versions/r1.5/performance/xla/operation_semantics#selectandscatter .
        ///
        /// ## Parameters
        ///
        /// |                           | Description                                                                                                                             |
        /// | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
        /// | `selection_function`      | The scalar function used to select between two values. Must take two arguments of type \f$E[]\f$ and return type \f$\mathit{Bool}[]\f$. |
        /// | `scatter_function`        | The scalar function used to apply a scattered value. Must take two arguments of type \f$E[]\f$ and return type \f$E[]\f$.               |
        /// | `window_shape`            | The shape \f$(w_1,\dots,w_n)\f$ of the selection window.                                                                                |
        /// | `window_movement_strides` | Movement strides \f$(s_1,\dots,s_n)\f$ to apply to the sliding window.                                                                  |
        ///
        /// ## Inputs
        ///
        /// |                | Type                                | Description                                                                                           |
        /// | -------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------- |
        /// | `arg_selectee` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$   | An input tensor of any shape, with the element type matching that expected by the selection function. |
        /// | `arg_source`   | \f$E[d'_1,\dots,d'_n]~(n \geq 0)\f$ | The input tensor from which to scatter values.                                                        |
        /// | `arg_init`     | \f$E[]\f$                           | A scalar to be used as an initial value in each output cell.                                          |
        ///
        /// ## Output
        ///
        /// | Type                   | Description          |
        /// | ---------------------- | -------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | See above algorithm. |
        class SelectAndScatter : public Op
        {
        public:
            /// \brief Constructs a select-and-scatter operation.
            ///
            /// \param arg_selectee The tensor view to be selected from.
            /// \param arg_source The tensor to scatter values from.
            /// \param arg_init The initial value for output.
            /// \param selection_function The selection function.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            SelectAndScatter(const std::shared_ptr<Node>& arg_selectee,
                             const std::shared_ptr<Node>& arg_source,
                             const std::shared_ptr<Node>& arg_init,
                             const std::shared_ptr<Function>& selection_function,
                             const std::shared_ptr<Function>& scatter_function,
                             const Shape& window_shape,
                             const Strides& window_movement_strides);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return A vector of length 2 containing the selection function as element 0, and the scatter function as element 1.
            std::vector<std::shared_ptr<Function>> get_functions() const override
            {
                return {m_selection_function, m_scatter_function};
            }
            /// \return The window shape.
            const Shape& get_window_shape() const { return m_window_shape; }
            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
        protected:
            std::shared_ptr<Function> m_selection_function;
            std::shared_ptr<Function> m_scatter_function;
            Shape m_window_shape;
            Strides m_window_movement_strides;
        };
    }
}
