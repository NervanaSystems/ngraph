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

#include <sstream>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "ngraph/runtime/plaidml/plaidml_build.hpp"
#include "ngraph/runtime/plaidml/plaidml_builder.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // PlaidML Operation implementation support.
            //
            // To add a new operation:
            //  1) Include the operation header
            //  2) Write the per-operation implementation definition
            //  3) Register the operation type, by instantiating Impl<op::OpClass>::Registration at
            //     global scope.
            //
            // Operation implementation definitions have access to all methods and member variables
            // of the general Impl template.

            // The accessor for the global operation handler map.
            std::unordered_map<std::type_index, std::function<void(Build*, const ngraph::Node&)>>*
                OpImplMap();

            // BaseImpl provides a context for operation interpretation, and provides a few useful
            // utility methods.
            template <typename O>
            class BaseImpl
            {
            public:
                BaseImpl(Build* build, const O& op)
                    : m_build{build}
                    , m_op{op}
                {
                }

            protected:
                Build* build() { return m_build; }
                const O& op() { return m_op; }
                // Returns the indicated operation input as a PlaidML variable.
                vertexai::plaidml::variable
                    op_input(std::size_t idx,
                             TensorContents as_contents = TensorContents::DATA) const
                {
                    const auto& ti = m_build->bindings.at(
                        m_op.get_inputs()[idx].get_output().get_tensor_ptr().get());
                    if (as_contents == TensorContents::DATA &&
                        ti.contents == TensorContents::LOGICAL)
                    {
                        return plaidml_logical_to_data(ti.var, m_build->config->debug);
                    }
                    return ti.var;
                }

                // Returns the 0th operation input as a PlaidML variable.
                vertexai::plaidml::variable op_input() const
                {
                    return op_input(0, TensorContents::DATA);
                }
                // Validates that the number of operation inputs matches the expected operation
                // input count.
                void check_inputs(std::size_t expected_input_count) const
                {
                    if (m_op.get_input_size() != expected_input_count)
                    {
                        std::ostringstream os;
                        os << "The PlaidML nGraph backend only supports " << m_op.description()
                           << " operations with an input count == " << expected_input_count
                           << " (got " << m_op.get_input_size() << " inputs)";
                        throw std::runtime_error{os.str()};
                    }
                }

                // Validates that the number of operation inputs is greater than or equal to the
                // expected operation input count.
                void check_inputs_ge(std::size_t minimum_input_count) const
                {
                    if (m_op.get_input_size() < minimum_input_count)
                    {
                        std::ostringstream os;
                        os << "The PlaidML nGraph backend only supports " << m_op.description()
                           << " operations with an input count >= " << minimum_input_count
                           << " (got " << m_op.get_input_size() << " inputs)";
                        throw std::runtime_error{os.str()};
                    }
                }

                // Validates that the number of operation outputs matches the expected operation
                // output count.
                void check_outputs(std::size_t expected_output_count) const
                {
                    if (m_op.get_output_size() != expected_output_count)
                    {
                        std::ostringstream os;
                        os << "The PlaidML nGraph backend only supports " << m_op.description()
                           << " operations with an output count == " << expected_output_count
                           << " (got " << m_op.get_output_size() << " outputs)";
                        throw std::runtime_error{os.str()};
                    }
                }

                // Binds the indicated operation output to the supplied PlaidML variable.
                void set_output(std::size_t idx,
                                vertexai::plaidml::variable var,
                                TensorContents contents = TensorContents::DATA)
                {
                    m_build->bindings.emplace(m_op.get_output_tensor_ptr(idx).get(),
                                              TensorInfo{std::move(var), contents});
                }

                // Binds the 0th operation output to the supplied PlaidML variable.
                void set_output(vertexai::plaidml::variable var,
                                TensorContents contents = TensorContents::DATA)
                {
                    m_build->bindings.emplace(m_op.get_output_tensor_ptr().get(),
                                              TensorInfo{std::move(var), contents});
                }

                // Gets a useful name for the current op.
                std::string get_op_name() const { return this->m_op.description(); }
                // Starts a Tile function builder.
                builder::Function start_tile_function() const
                {
                    return builder::Function{get_op_name(), m_build->config->debug};
                }

            private:
                Build* m_build;
                const O& m_op;
            };

            // ParentImpl sets the base implementation class for a particular operation class; the
            // Impl template uses this to figure out which class to derive from when implementing a
            // particular operation.  This is meant to be specialized as needed.
            template <typename O>
            struct ParentImpl
            {
                using Type = BaseImpl<O>;
            };

            // Impl is the common operation implementation class.  It declares an operator(), to be
            // subsequently defined with the implementation for the particular operation.
            //
            // Operations that do require extensions may derive their common class from BaseImpl,
            // and pass it to the Impl template.  Alternatively, they may specialize the Impl
            // template, replacing it with their own implementation.
            template <typename O>
            class Impl : public ParentImpl<O>::Type
            {
            public:
                Impl(Build* build, const O& op)
                    : ParentImpl<O>::Type{build, op}
                {
                }
                void operator()();

                static void handler(Build* build, const ngraph::Node& node)
                {
                    Impl<O>(build, dynamic_cast<const O&>(node))();
                }

                // Registration handles the registration of a particular operation implementation.
                // To use it, instantiate a variable of type Impl<op::OpClass>::Registration at
                // global scope.
                class Registration
                {
                public:
                    Registration()
                    {
                        OpImplMap()->emplace(std::type_index{typeid(O)}, &Impl<O>::handler);
                    }
                };
            };
        }
    }
}
