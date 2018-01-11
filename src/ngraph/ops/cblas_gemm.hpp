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

#include "ngraph/node.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/util.hpp"

#include <memory>

namespace ngraph
{
    namespace op
    {
        class CblasGemm : public RequiresTensorViewArgs
        {
        public:
            CblasGemm(std::shared_ptr<Node> W,
                      std::shared_ptr<Node> x,
                      std::shared_ptr<Node> b,
                      Shape shape_w,
                      Shape shape_x,
                      bool transpose_w,
                      bool transpose_x)
                : RequiresTensorViewArgs("CblassGemm", {W, x, b})
                , m_shape_w(shape_w)
                , m_shape_x(shape_x)
                , m_transpose_w(transpose_w)
                , m_transpose_x(transpose_x)

            {
                if (shape_w.size() != 2)
                {
                    //
                    NGRAPH_DEBUG << "W shape = " << vector_to_string(shape_w);
                    throw "W.shape != 2";
                }

                if (shape_x.size() != 2)
                {
                    //
                    NGRAPH_DEBUG << "x shape = " << vector_to_string(shape_x);
                    throw "x.shape != 2";
                }

                size_t dot_dimension_w = (transpose_w) ? 0 : 1;
                size_t dot_dimension_x = (transpose_x) ? 1 : 0;

                NGRAPH_DEBUG << "dot_dimension_w = " << dot_dimension_w
                             << " , dot_dimension_x = " << dot_dimension_x;
                NGRAPH_DEBUG << "W shape = " << vector_to_string(shape_w)
                             << " , x shape = " << vector_to_string(shape_x);

                if (shape_w.at(dot_dimension_w) != shape_x.at(dot_dimension_x))
                {
                    throw "product dimensions are not equal";
                }

                auto dot_shape =
                    Shape{shape_w.at(1 - dot_dimension_w), shape_x.at(1 - dot_dimension_x)};
                NGRAPH_DEBUG << "dot_shape shape = " << vector_to_string(dot_shape)
                             << " , b shape = " << vector_to_string(b->get_shape());

                add_output(W->get_element_type(), dot_shape);
            }

            bool get_is_arg0_transposed() const { return m_transpose_w; }
            bool get_is_arg1_transposed() const { return m_transpose_x; }
            Shape get_arg0_shape() const { return m_shape_w; }
            Shape get_arg1_shape() const { return m_shape_x; }
            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 2)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<CblasGemm>(new_args.at(0),
                                                   new_args.at(1),
                                                   new_args.at(1),
                                                   m_shape_w,
                                                   m_shape_x,
                                                   m_transpose_w,
                                                   m_transpose_x);
            }

        private:
            Shape m_shape_w;
            Shape m_shape_x;
            bool m_transpose_w;
            bool m_transpose_x;
        };
    }
}
