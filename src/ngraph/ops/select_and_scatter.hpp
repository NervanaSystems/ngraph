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

#pragma once

#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Select-and-scatter operation.
        ///
        /// TODO: More formal definition. For now, see: https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter.
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
        /// | Type                     | Description                            |
        /// | ------------------------ | -------------------------------------- |
        /// | \f$E[d'_1,\dots,d'_n]\f$ | (TODO: explain more) See the XLA docs. |
        class SelectAndScatter : public RequiresTensorViewArgs
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

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 3)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<SelectAndScatter>(new_args.at(0),
                                                          new_args.at(1),
                                                          new_args.at(2),
                                                          m_selection_function,
                                                          m_scatter_function,
                                                          m_window_shape,
                                                          m_window_movement_strides);
            }

            /// \return A vector of length 2 containing the selection function as element 0, and the scatter function as element 1.
            std::vector<std::shared_ptr<Function>> get_functions() const override
            {
                return {m_selection_function, m_scatter_function};
            }
            /// \return The window shape.
            const Shape& get_window_shape() const { return m_window_shape; }
            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            bool is_functionally_identical(const Node&) const override;

        protected:
            std::shared_ptr<Function> m_selection_function;
            std::shared_ptr<Function> m_scatter_function;
            Shape m_window_shape;
            Strides m_window_movement_strides;
        };
    }
}
