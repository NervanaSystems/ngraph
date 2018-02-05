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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Batched convolution operation, with optional window dilation and stride.
        ///
        /// Convolution takes two inputs:
        ///
        /// 1. <i>(the idata batch)</i> a tensor of shape \f$(N,C_\textit{in},d_1,\dots,d_n)\f$ where \f$n > 0\f$, every \f$d_i > 0\f$, and where \f$N\f$ is the batch size
        ///    (number of data items) and \f$C_\textit{in} > 0\f$ is the number of input channels (sometimes called features); and
        /// 2. <i>(the filters)</i> a tensor of shape \f$(C_\textit{out},C_\textit{in},d^f_1,\dots,d^f_n)\f$, where \f$C_\textit{out} > 0\f$ is the number of output channels
        ///    (sometimes called features) and \f$(d^f_1,\dots,d^f_n)\f$ are the filter dimensions. It is required that for all \f$i\f$, \f$0 < l_i(d^f_i - 1) + 1 \le (d_i - 1)*g_i + 1\f$.
        ///    (See below for the definition of the window dilation \f$l_i\f$ and the data dilation \f$t_i\f$);
        ///
        /// and five optional parameters:
        ///
        /// 3. <i>(the window movement strides)</i> a vector of positive integers \f$(s_1,\dots,s_n)\f$ (default is all ones),
        /// 4. <i>(the window dilation strides)</i> a vector of positive integers \f$(l_1,\dots,l_n)\f$ (default is all ones),
        /// 5. <i>(the padding below)</i> a vector of (possibly negative) integers \f$(p_1,\dots,p_n)\f$ (default is all zeros),
        /// 6. <i>(the padding above)</i> a vector of (possibly negative) integers \f$(q_1,\dots,q_n)\f$ (default is all zeros), and
        /// 7. <i>(the data dilation strides)</i> a vector of non-negative integers \f$(q_1,\dots,q_n)\f$ (default is all ones).
        ///
        /// The output has the shape \f$(N,C_\textit{out},d'_1,\dots,d'_n)\f$, where \f$d'_n = \lceil \frac{(d_i - 1) * t_i + 1 + p_i + q_i - l_i(d^f_i - 1)}{s_i} \rceil\f$.
        ///
        /// Given an input data batch tensor \f$T_\textit{in}\f$, first define the <i>transformed input tensor</i> \f$T_\textit{trans}\f$, with shape \f$(N,C_\textit{in},(d_1 - 1)*t_1+1+p_1+q_1,\dots,(d_n - 1)*t_n+1+p_n+q_n)\f$, as follows:
        ///
        /// \f[
        ///      T_\textit{trans}[a,c,i_1,\dots,i_n] = T[a,c,(i_1 - p_1)/t_1,\dots,(i_n - p_n)/t_n] \text{ if for all }k, t_k evenly divides (i_k - p_k) and p_k \le i_k \lt p_k + (d_k - 1)*t_k + 1, \text{ else } 0
        /// \f]
        ///
        /// then, given an input filter tensor \f$T_\textit{filt}\f$, the output tensor \f$T_\textit{out}\f$ is defined by the equation.
        ///
        /// \f[
        ///      T_\textit{out}[a,c_\textit{out},i_1,\dots,i_n] = \sum_{c_\textit{in}=0,j_1=0,\dots,j_n=0}^{c_\textit{in}=C_\textit{in}-1,j_1=d^f_1-1,\dots,j_n=d^f_n-1} (T_\textit{filt}[c_\textit{out},c_\textit{in},j_1,\dots,j_n] \cdot T_\textit{trans}[a,c_\textit{in},s_1i_1+l_1j_1,\dots,s_ni_n+l_nj_n])
        /// \f]
        ///
        class Convolution : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched convolution operation.
            ///
            /// \param data_batch The node producing the input data batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            /// \param window_dilation_strides The window dilation strides.
            /// \param padding_below The padding-below sizes.
            /// \param padding_above The padding-above sizes.
            /// \param data_dilation_strides The data dilation strides.
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides,
                        const CoordinateDiff& padding_below,
                        const CoordinateDiff& padding_above,
                        const Strides& data_dilation_strides);

            /// \brief Constructs a batched convolution operation with no data dilation (i.e., all data dilation strides are 1).
            ///
            /// \param data_batch The node producing the input data batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            /// \param window_dilation_strides The window dilation strides.
            /// \param padding_below The padding-below sizes.
            /// \param padding_above The padding-above sizes.
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides,
                        const CoordinateDiff& padding_below,
                        const CoordinateDiff& padding_above);

            /// \brief Constructs a batched convolution operation with no padding or data dilation (i.e., padding above and below are 0 everywhere, and all data dilation strides are 1).
            ///
            /// \param data_batch The node producing the input data batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            /// \param window_dilation_strides The window dilation strides.
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides);

            /// \brief Constructs a batched convolution operation with no window dilation, padding, or data dilation (i.e., padding above and below are 0 everywhere, and all window/data dilation strides are 1).
            ///
            /// \param data_batch The node producing the input data batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides);

            /// \brief Constructs a batched convolution operation with no window dilation or movement stride (i.e., padding above and below are 0 everywhere, and all window/data dilation strides and window movement strides are 1).
            ///
            /// \param data_batch The node producing the input data batch tensor.
            /// \param filters The node producing the filters tensor.
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters);

            virtual std::shared_ptr<Node> copy_with_new_args(const Nodes& new_args) const override;
            void generate_adjoints(autodiff::Adjoints& adjoints,
                                   const std::shared_ptr<Node>& delta) override;

            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            /// \return The window dilation strides.
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            /// \return The padding-below sizes (possibly negative).
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            /// \return The padding-above sizes (possibly negative).
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            /// \return The input data dilation strides.
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;

        private:
            static Strides default_strides(const std::shared_ptr<Node>& data_batch);
            static CoordinateDiff default_padding(const std::shared_ptr<Node>& data_batch);
        };

        /// \brief Data batch backprop for batched convolution operation.
        class ConvolutionBackpropData : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched-convolution data batch-backprop operation.
            ///
            /// \param data_batch_shape The shape of the data batch from forward-prop.
            /// \param filters The node producing the filters from forward-prop.
            /// \param output_delta The node producing output delta.
            /// \param window_movement_strides_forward The window movement strides from forward-prop.
            /// \param window_dilation_strides_forward The window dilation strides from forward-prop.
            /// \param padding_below_forward The padding-below sizes from forward-prop.
            /// \param padding_above_forward The padding-above sizes from forward-prop.
            /// \param data_dilation_strides_forward The data dilation strides from forward-prop.
            ConvolutionBackpropData(const Shape& data_batch_shape,
                                    const std::shared_ptr<Node>& filters,
                                    const std::shared_ptr<Node>& output_delta,
                                    const Strides& window_movement_strides_forward,
                                    const Strides& window_dilation_strides_forward,
                                    const CoordinateDiff& padding_below_forward,
                                    const CoordinateDiff& padding_above_forward,
                                    const Strides& data_dilation_strides_forward);

            virtual std::shared_ptr<Node> copy_with_new_args(const Nodes& new_args) const override;

            /// \return The data batch shape.
            const Shape& get_data_batch_shape() const { return m_data_batch_shape; }
            /// \return The window movement strides from the forward prop.
            const Strides& get_window_movement_strides_forward() const
            {
                return m_window_movement_strides_forward;
            }
            /// \return The window dilation strides from the forward prop.
            const Strides& get_window_dilation_strides_forward() const
            {
                return m_window_dilation_strides_forward;
            }
            /// \return The padding-below sizes (possibly negative) from the forward prop.
            const CoordinateDiff& get_padding_below_forward() const
            {
                return m_padding_below_forward;
            }
            /// \return The padding-above sizes (possibly negative) from the forward prop.
            const CoordinateDiff& get_padding_above_forward() const
            {
                return m_padding_above_forward;
            }
            /// \return The input data dilation strides from the forward prop.
            const Strides& get_data_dilation_strides_forward() const
            {
                return m_data_dilation_strides_forward;
            }

            /// \return The window movement strides for the backward prop.
            const Strides& get_window_movement_strides_backward() const
            {
                return m_window_movement_strides_backward;
            }
            /// \return The window dilation strides for the backward prop.
            const Strides& get_window_dilation_strides_backward() const
            {
                return m_window_dilation_strides_backward;
            }
            /// \return The padding-below sizes (possibly negative) for the backward prop.
            const CoordinateDiff& get_padding_below_backward() const
            {
                return m_padding_below_backward;
            }
            /// \return The padding-above sizes (possibly negative) for the backward prop.
            const CoordinateDiff& get_padding_above_backward() const
            {
                return m_padding_above_backward;
            }
            /// \return The input data dilation strides for the backward prop.
            const Strides& get_data_dilation_strides_backward() const
            {
                return m_data_dilation_strides_backward;
            }

        protected:
            Shape m_data_batch_shape;
            Strides m_window_movement_strides_forward;
            Strides m_window_dilation_strides_forward;
            CoordinateDiff m_padding_below_forward;
            CoordinateDiff m_padding_above_forward;
            Strides m_data_dilation_strides_forward;

            Strides m_window_movement_strides_backward;
            Strides m_window_dilation_strides_backward;
            CoordinateDiff m_padding_below_backward;
            CoordinateDiff m_padding_above_backward;
            Strides m_data_dilation_strides_backward;
        };

        /// \brief Filters backprop for batched convolution operation.
        class ConvolutionBackpropFilters : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched-convolution filter-backprop operation.
            ///
            /// \param data_batch The tensor producing the data batch from forward-prop.
            /// \param filters_shape The shape of the filters from forward-prop.
            /// \param output_delta The node producing output delta.
            /// \param window_movement_strides_forward The window movement strides from forward-prop.
            /// \param window_dilation_strides_forward The window dilation strides from forward-prop.
            /// \param padding_below_forward The padding-below sizes from forward-prop.
            /// \param padding_above_forward The padding-above sizes from forward-prop.
            /// \param data_dilation_strides_forward The data dilation strides from forward-prop.
            ConvolutionBackpropFilters(const std::shared_ptr<Node>& data_batch,
                                       const Shape& filters_shape,
                                       const std::shared_ptr<Node>& output_delta,
                                       const Strides& window_movement_strides_forward,
                                       const Strides& window_dilation_strides_forward,
                                       const CoordinateDiff& padding_below_forward,
                                       const CoordinateDiff& padding_above_forward,
                                       const Strides& data_dilation_strides_forward);

            virtual std::shared_ptr<Node> copy_with_new_args(const Nodes& new_args) const override;

            /// \return The filters tensor shape.
            const Shape& get_filters_shape() const { return m_filters_shape; }
            /// \return The window movement strides from the forward prop.
            const Strides& get_window_movement_strides_forward() const
            {
                return m_window_movement_strides_forward;
            }
            /// \return The window dilation strides from the forward prop.
            const Strides& get_window_dilation_strides_forward() const
            {
                return m_window_dilation_strides_forward;
            }
            /// \return The padding-below sizes (possibly negative) from the forward prop.
            const CoordinateDiff& get_padding_below_forward() const
            {
                return m_padding_below_forward;
            }
            /// \return The padding-above sizes (possibly negative) from the forward prop.
            const CoordinateDiff& get_padding_above_forward() const
            {
                return m_padding_above_forward;
            }
            /// \return The data dilation strides from the forward prop.
            const Strides& get_data_dilation_strides_forward() const
            {
                return m_data_dilation_strides_forward;
            }

            /// \return The window movement strides for the backward prop.
            const Strides& get_window_movement_strides_backward() const
            {
                return m_window_movement_strides_backward;
            }
            /// \return The window dilation strides for the backward prop.
            const Strides& get_window_dilation_strides_backward() const
            {
                return m_window_dilation_strides_backward;
            }
            /// \return The padding-below sizes (possibly negative) for the backward prop.
            const CoordinateDiff& get_padding_below_backward() const
            {
                return m_padding_below_backward;
            }
            /// \return The padding-above sizes (possibly negative) for the backward prop.
            const CoordinateDiff& get_padding_above_backward() const
            {
                return m_padding_above_backward;
            }
            /// \return The data dilation strides for the backward prop.
            const Strides& get_data_dilation_strides_backward() const
            {
                return m_data_dilation_strides_backward;
            }

        protected:
            Shape m_filters_shape;
            Strides m_window_movement_strides_forward;
            Strides m_window_dilation_strides_forward;
            CoordinateDiff m_padding_below_forward;
            CoordinateDiff m_padding_above_forward;
            Strides m_data_dilation_strides_forward;

            Strides m_window_movement_strides_backward;
            Strides m_window_dilation_strides_backward;
            CoordinateDiff m_padding_below_backward;
            CoordinateDiff m_padding_above_backward;
            Strides m_data_dilation_strides_backward;
        };
    }
}
