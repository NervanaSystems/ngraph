/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <plaidml/plaidml++.h>

#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "ngraph/runtime/plaidml/plaidml_config.hpp"

// Utilities for constructing PlaidML functions.

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            namespace builder
            {
                class BinaryContraction;
                class ContractionInput;
                class ContractionOutput;
                class Elementwise;
                class Function;
                class Input;
                class Output;
                class Statement;
                class TernaryContraction;
                class UnaryContraction;
            }
        }
    }
}

// Function provides a fluent interface for the construction of the text form of PlaidML functions.
// It's intended to be simpler to use and produce simpler code than using direct string construction.
//
// N.B. It's possible to capture the intermediate pieces as they're being added to a function
// (e.g. in order to directly code loops which call methods on them), but it's important to
// remember that what's returned are references, not objects; the caller is responsible for
// instantiating a Function instance and keeping it alive as long as there are any outstanding
// references to its constituent pieces.
class ngraph::runtime::plaidml::builder::Function final
{
public:
    Function(const std::string& name, bool debug);
    Function(const Function&) = delete;
    Function& operator=(const Function&) = delete;
    Function(Function&&) = default;
    Function& operator=(Function&&) = default;

    // Builds the final string form of the function.
    std::string to_string() const;

    // Finalizes a function, transforming it into a PlaidML function application object.
    vertexai::plaidml::application finalize() const;

    // Adds an input to the function.
    Function& add(Input input) &;
    Function&& add(Input input) &&;

    // Adds an output to the function.
    Function& add(Output output) &;
    Function&& add(Output output) &&;

    // Adds a contraction to the function.
    Function& add(TernaryContraction contraction) &;
    Function&& add(TernaryContraction contraction) &&;
    Function& add(BinaryContraction contraction) &;
    Function&& add(BinaryContraction contraction) &&;
    Function& add(UnaryContraction contraction) &;
    Function&& add(UnaryContraction contraction) &&;

    // Adds an elementwise mapping to the function.
    Function& add(Elementwise elementwise) &;
    Function&& add(Elementwise elementwise) &&;

private:
    std::string name_;
    bool debug_;
    std::list<Input> inputs_;
    std::list<Output> outputs_;
    std::list<std::unique_ptr<Statement>> stmts_;
};

// Input represents an input being added to a function.
class ngraph::runtime::plaidml::builder::Input final
{
public:
    Input(vertexai::plaidml::variable var, std::string name);

    // Adds a list of dimensions to the input, [first..limit).
    Input& add_dims(std::string prefix, std::size_t first, std::size_t limit) &;
    Input&& add_dims(std::string prefix, std::size_t first, std::size_t limit) &&;

    // Adds a list of dimensions to the input, [first..limit), in reverse order.
    Input& add_rdims(std::string prefix, std::size_t limit, std::size_t first) &;
    Input&& add_rdims(std::string prefix, std::size_t limit, std::size_t first) &&;

    // Adds a fixed list of dimensions to the input.
    Input& add_dims(std::initializer_list<std::string> s) &;
    Input&& add_dims(std::initializer_list<std::string> s) &&;

    // Adds dimensions by passing an insert iterator to a lambda.
    template <typename L>
    Input& add_dims(L lambda) &
    {
        lambda(std::back_inserter(dims_));
        return *this;
    }

    template <typename L>
    Input&& add_dims(L lambda) &&
    {
        lambda(std::back_inserter(dims_));
        return std::move(*this);
    }

private:
    friend class Function;

    vertexai::plaidml::variable var_;
    std::string name_;
    std::list<std::string> dims_;
};

// Output represents an output being added to a function.
class ngraph::runtime::plaidml::builder::Output final
{
public:
    Output(std::string name);

private:
    friend class Function;

    std::string name_;
};

// Statement is the abstract base class for UnaryContraction, BinaryContraction,
// TernaryContraction, and Elementwise objects.
class ngraph::runtime::plaidml::builder::Statement
{
public:
    virtual ~Statement() = default;

protected:
    Statement() = default;
    Statement(const Statement&) = default;
    Statement(Statement&&) = default;
};

// Elementwise represents an elementwise mapping being added to a function.
class ngraph::runtime::plaidml::builder::Elementwise final : public Statement
{
public:
    Elementwise(std::string lhs, std::string rhs);

private:
    friend class Function;

    std::string lhs_;
    std::string rhs_;
};

// The output of a contraction
class ngraph::runtime::plaidml::builder::ContractionOutput final
{
public:
    explicit ContractionOutput(std::string name);

    ContractionOutput& add_indices(std::string prefix, std::size_t first, std::size_t limit) &;
    ContractionOutput&& add_indices(std::string prefix, std::size_t first, std::size_t limit) &&;
    ContractionOutput& add_rindices(std::string prefix, std::size_t limit, std::size_t first) &;
    ContractionOutput&& add_rindices(std::string prefix, std::size_t limit, std::size_t first) &&;
    ContractionOutput& add_indices(std::initializer_list<std::string> s) &;
    ContractionOutput&& add_indices(std::initializer_list<std::string> s) &&;

    template <typename L>
    ContractionOutput& add_indices(L lambda) &
    {
        lambda(std::back_inserter(indices_));
        return *this;
    }

    template <typename L>
    ContractionOutput&& add_indices(L lambda) &&
    {
        lambda(std::back_inserter(indices_));
        return std::move(*this);
    }

    ContractionOutput& add_dims(std::string prefix, std::size_t first, std::size_t limit) &;
    ContractionOutput&& add_dims(std::string prefix, std::size_t first, std::size_t limit) &&;
    ContractionOutput& add_rdims(std::string prefix, std::size_t limit, std::size_t first) &;
    ContractionOutput&& add_rdims(std::string prefix, std::size_t limit, std::size_t first) &&;
    ContractionOutput& add_dims(std::initializer_list<std::string> s) &;
    ContractionOutput&& add_dims(std::initializer_list<std::string> s) &&;

    template <typename L>
    ContractionOutput& add_dims(L lambda) &
    {
        lambda(std::back_inserter(dims_));
        return *this;
    }

    template <typename L>
    ContractionOutput&& add_dims(L lambda) &&
    {
        lambda(std::back_inserter(dims_));
        return std::move(*this);
    }

private:
    friend class Function;

    std::string name_;
    std::list<std::string> indices_;
    std::list<std::string> dims_;
};

// An input to a contraction
class ngraph::runtime::plaidml::builder::ContractionInput final
{
public:
    explicit ContractionInput(std::string name)
        : name_{std::move(name)}
    {
    }

    ContractionInput& add_indices(std::string prefix, std::size_t first, std::size_t limit) &;
    ContractionInput&& add_indices(std::string prefix, std::size_t first, std::size_t limit) &&;
    ContractionInput& add_rindices(std::string prefix, std::size_t limit, std::size_t first) &;
    ContractionInput&& add_rindices(std::string prefix, std::size_t limit, std::size_t first) &&;
    ContractionInput& add_indices(std::initializer_list<std::string> s) &;
    ContractionInput&& add_indices(std::initializer_list<std::string> s) &&;

    template <typename L>
    ContractionInput& add_indices(L lambda) &
    {
        lambda(std::back_inserter(indices_));
        return *this;
    }

    template <typename L>
    ContractionInput&& add_indices(L lambda) &&
    {
        lambda(std::back_inserter(indices_));
        return std::move(*this);
    }

private:
    friend class Function;

    std::string name_;
    std::list<std::string> indices_;
};

// UnaryContraction represents a unary contraction being added to a function.
class ngraph::runtime::plaidml::builder::UnaryContraction final : public Statement
{
public:
    explicit UnaryContraction(std::string agg_op);

    UnaryContraction& set(ContractionInput input) &;
    UnaryContraction&& set(ContractionInput input) &&;
    UnaryContraction& set(ContractionOutput output) &;
    UnaryContraction&& set(ContractionOutput output) &&;
    UnaryContraction& set_default(std::string tensor) &;
    UnaryContraction&& set_default(std::string tensor) &&;

    template <typename L>
    UnaryContraction& add_constraints(L lambda) &
    {
        lambda(std::back_inserter(constraints_));
        return *this;
    }
    template <typename L>
    UnaryContraction&& add_constraints(L lambda) &&
    {
        lambda(std::back_inserter(constraints_));
        return std::move(*this);
    }

private:
    friend class Function;

    std::string agg_op_;
    std::list<std::string> constraints_;
    std::unique_ptr<ContractionOutput> output_;
    std::unique_ptr<ContractionInput> input_;
    std::string default_;
};

// BinaryContraction represents a binary contraction being added to a function.
class ngraph::runtime::plaidml::builder::BinaryContraction final : public Statement
{
public:
    BinaryContraction(std::string agg_op, std::string comb_op);

    BinaryContraction& set_lhs(ContractionInput input) &;
    BinaryContraction&& set_lhs(ContractionInput input) &&;
    BinaryContraction& set_rhs(ContractionInput input) &;
    BinaryContraction&& set_rhs(ContractionInput input) &&;
    BinaryContraction& set(ContractionOutput output) &;
    BinaryContraction&& set(ContractionOutput output) &&;
    BinaryContraction& set_default(std::string tensor) &;
    BinaryContraction&& set_default(std::string tensor) &&;

    template <typename L>
    BinaryContraction& add_constraints(L lambda) &
    {
        lambda(std::back_inserter(constraints_));
        return *this;
    }
    template <typename L>
    BinaryContraction&& add_constraints(L lambda) &&
    {
        lambda(std::back_inserter(constraints_));
        return std::move(*this);
    }

private:
    friend class Function;

    std::string agg_op_;
    std::string comb_op_;
    std::list<std::string> constraints_;
    std::unique_ptr<ContractionOutput> output_;
    std::unique_ptr<ContractionInput> lhs_;
    std::unique_ptr<ContractionInput> rhs_;
    std::string default_;
};

// TernaryContraction represents a ternary contraction being added to a function
class ngraph::runtime::plaidml::builder::TernaryContraction final : public Statement
{
public:
    TernaryContraction(std::string agg_op, std::string comb_op);

    TernaryContraction& set_first(ContractionInput input) &;
    TernaryContraction&& set_first(ContractionInput input) &&;
    TernaryContraction& set_second(ContractionInput input) &;
    TernaryContraction&& set_second(ContractionInput input) &&;
    TernaryContraction& set_third(ContractionInput input) &;
    TernaryContraction&& set_third(ContractionInput input) &&;
    TernaryContraction& set(ContractionOutput output) &;
    TernaryContraction&& set(ContractionOutput output) &&;

    template <typename L>
    TernaryContraction& add_constraints(L lambda) &
    {
        lambda(std::back_inserter(constraints_));
        return *this;
    }
    template <typename L>
    TernaryContraction&& add_constraints(L lambda) &&
    {
        lambda(std::back_inserter(constraints_));
        return std::move(*this);
    }

private:
    friend class Function;

    std::string agg_op_;
    std::string comb_op_;
    std::list<std::string> constraints_;
    std::unique_ptr<ContractionOutput> output_;
    std::unique_ptr<ContractionInput> first_;
    std::unique_ptr<ContractionInput> second_;
    std::unique_ptr<ContractionInput> third_;
};
