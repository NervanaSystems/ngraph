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

#include <limits>
#include <memory>

#include <iostream>

#include "ngraph/node.hpp"
#include "ngraph/ops/util/requires_tensor_view_args.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            static void default_printer(const std::string data, const std::string name)
            {
                std::cerr << name << "'s values are " << data << std::endl;
            };

            class Trace : public RequiresTensorViewArgs
            {
            public:
                typedef void (*TraceCallback)(std::string data, std::string name);

                Trace(const std::shared_ptr<Node>& arg,
                      TraceCallback callback = default_printer,
                      size_t from = 0,
                      size_t to = std::numeric_limits<size_t>::max())
                    : RequiresTensorViewArgs("Trace", {arg})
                    , m_callback(callback)
                    , m_from(from)
                    , m_to(to)
                {
                    set_value_type_checked(arg->get_element_type(), arg->get_shape());
                }

                virtual std::shared_ptr<Node> copy_with_new_args(
                    const std::vector<std::shared_ptr<Node>>& new_args) const override
                {
                    if (new_args.size() != 1)
                    {
                        throw ngraph_error("Incorrect number of new arguments");
                    }
                    return std::make_shared<op::util::Trace>(new_args.at(0));
                }

                TraceCallback get_callback() const { return m_callback; }
                size_t get_from() const { return m_from; }
                size_t get_to() const { return m_to; }
            private:
                TraceCallback m_callback;
                size_t m_from;
                size_t m_to;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const std::shared_ptr<Node>& delta) override
                {
                    adjoints.add_delta(get_input_op(0), delta);
                }
            };
        }
    }
}
