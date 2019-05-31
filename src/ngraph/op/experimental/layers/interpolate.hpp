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

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        typedef struct
        {
            int height = -1;
            int width = -1;
            float zoom_factor = 0;
            float shrink_factor = 0;
            float scale_factor = 1.0;
            bool align_coreners = true;
            int pad_beg = 0;
            int pad_end = 0;
        } InterpolateAttrs;

        /// \brief Layer which performs bilinear interpolation
        class Interpolate : public Op
        {
        public:
            /// \brief Constructs a Interpolate operation
            ///
            /// \param image	Input image
            /// \param attrs    Interpolation attributes
            Interpolate(const std::shared_ptr<Node>& image, const InterpolateAttrs& attrs);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const InterpolateAttrs& get_attrs() const { return m_attrs; }
        private:
            InterpolateAttrs m_attrs;
        };

        /// \brief Layer which performs bilinear interpolation
        class DynInterpolate : public Op
        {
        public:
            /// \brief Constructs a DynInterpolate operation
            ///
            /// \param image	    Input image
            /// \param output_shape Output shape
            /// \param attrs        Interpolation attributes
            DynInterpolate(const std::shared_ptr<Node>& image,
                           const std::shared_ptr<Node>& output_shape,
                           const InterpolateAttrs& attrs);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const InterpolateAttrs& get_attrs() const { return m_attrs; }
        private:
            InterpolateAttrs m_attrs;
        };
    }
}
