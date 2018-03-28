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

#include <memory>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        class CallFrame;
    }

    namespace xla
    {
        namespace op
        {
            /// A special Op for constructing graphs with XLA tuples.
            /// Can only be used as an argument to the get_tuple_element function, which returns the node
            /// that was used when the tuple was constructed; the constructed graph should have no Tuple
            /// nodes in it.
            class Tuple : public Node
            {
            public:
                Tuple(const NodeVector& nodes);

                std::shared_ptr<Node> get_tuple_element(size_t i);
                size_t get_tuple_size() const;
                const NodeVector& get_elements() const;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

            protected:
                NodeVector m_elements;
            };

            std::shared_ptr<Node> get_tuple_element(std::shared_ptr<Node> tuple, size_t i);
        }

        /// Extends functions to let results include xla::op::Tuple, and paramaters to include
        /// xla::op::Tuple of op::Parameter trees.
        class XLAFunction : public Function
        {
        public:
            XLAFunction(const NodeVector& results,
                        const NodeVector& parameters,
                        const std::string& name = "");
        };

        using XLAValues = std::vector<std::shared_ptr<runtime::TensorView>>;

        /// An XLATuple is a implemented as an extension of a float scalar so that it fits in nicely
        /// with the nGraph type hierarchy.
        class XLATuple : public runtime::TensorView
        {
        public:
            XLATuple(const XLAValues& elements);

            const ngraph::runtime::TensorViewPtrs& get_elements() const;
            std::shared_ptr<runtime::TensorView> get_tuple_element(size_t i) const;
            size_t get_tuple_size() const;
            virtual void write(const void* p, size_t tensor_offset, size_t n) override;
            virtual void read(void* p, size_t tensor_offset, size_t n) const override;

        protected:
            std::vector<std::shared_ptr<runtime::TensorView>> m_elements;
        };

        /// Convenience function for making a runtime tuple.
        inline std::shared_ptr<XLATuple> make_tuple(const XLAValues& elements)
        {
            return std::make_shared<XLATuple>(elements);
        }

        /// Convenience function for accessing a tuple element.
        std::shared_ptr<runtime::TensorView> get_tuple_element(std::shared_ptr<XLATuple> xla_tuple,
                                                               size_t i);

        /// Invoke a call frame where some arguments might be XLATuples
        void call(std::shared_ptr<runtime::CallFrame> call_frame,
                  const ngraph::runtime::TensorViewPtrs& outputs,
                  const ngraph::runtime::TensorViewPtrs& inputs);
    }
}
