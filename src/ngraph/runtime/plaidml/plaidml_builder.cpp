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

#include <sstream>
#include <stdexcept>
#include <utility>

#include "ngraph/runtime/plaidml/plaidml_builder.hpp"
#include "ngraph/runtime/plaidml/plaidml_logger.hpp"

namespace vp = vertexai::plaidml;

ngraph::runtime::plaidml::builder::Function::Function(const std::string& name, bool debug)
    : m_name{name}
    , m_debug{debug}
{
}

std::string ngraph::runtime::plaidml::builder::Function::to_string() const
{
    std::ostringstream s;
    s << "function (";
    bool first = true;
    for (const auto& input : m_inputs)
    {
        if (!first)
        {
            s << ", ";
        }
        first = false;
        s << input.m_name;
        if (input.m_dims.size())
        {
            s << "[";
            bool first_dim = true;
            for (const auto& dim : input.m_dims)
            {
                if (!first_dim)
                {
                    s << ", ";
                }
                first_dim = false;
                s << dim;
            }
            s << "]";
        }
    }
    s << ") -> (";
    first = true;
    for (const auto& output : m_outputs)
    {
        if (!first)
        {
            s << ", ";
        }
        first = false;
        s << output.m_name;
    }
    s << ") {\n";
    std::string name_annotation;
    if (m_name.size())
    {
        name_annotation = "[[name(op" + m_name + ")]]\n  ";
    }
    for (const std::unique_ptr<Statement>& stmt : m_stmts)
    {
        s << "  " << name_annotation;
        {
            const TernaryContraction* tc = dynamic_cast<const TernaryContraction*>(stmt.get());
            if (tc)
            {
                if (!tc->m_output || !tc->m_first || !tc->m_second || !tc->m_third)
                {
                    throw std::logic_error{"Incomplete contraction"};
                }
                if (tc->m_output->m_indices.size() != tc->m_output->m_dims.size())
                {
                    throw std::logic_error{"Contraction index count != dimension count"};
                }
                s << tc->m_output->m_name << "[";
                first = true;
                for (const auto& idx : tc->m_output->m_indices)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                if (tc->m_output->m_indices.size())
                {
                    s << " : ";
                }
                first = true;
                for (const auto& dim : tc->m_output->m_dims)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << dim;
                }
                s << "] = " << tc->m_agg_op << "(" << tc->m_first->m_name << "[";
                first = true;
                for (const auto& idx : tc->m_first->m_indices)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "] == " << tc->m_second->m_name << "[";
                first = true;
                for (const auto& idx : tc->m_second->m_indices)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "] " << tc->m_comb_op << " " << tc->m_third->m_name << "[";
                first = true;
                for (const auto& idx : tc->m_third->m_indices)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "])";
                for (const auto& constraint : tc->m_constraints)
                {
                    s << ", " << constraint;
                }
                s << ";\n";
                continue;
            }
            const BinaryContraction* bc = dynamic_cast<const BinaryContraction*>(stmt.get());
            if (bc)
            {
                if (!bc->m_output || !bc->m_lhs || !bc->m_rhs)
                {
                    throw std::logic_error{"Incomplete contraction"};
                }
                if (bc->m_output->m_indices.size() != bc->m_output->m_dims.size())
                {
                    throw std::logic_error{"Contraction index count != dimension count"};
                }
                s << bc->m_output->m_name << "[";
                first = true;
                for (const auto& idx : bc->m_output->m_indices)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                if (bc->m_output->m_indices.size())
                {
                    s << " : ";
                }
                first = true;
                for (const auto& dim : bc->m_output->m_dims)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << dim;
                }
                s << "] = " << bc->m_agg_op << "(" << bc->m_lhs->m_name << "[";
                first = true;
                for (const auto& idx : bc->m_lhs->m_indices)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "] " << bc->m_comb_op << " " << bc->m_rhs->m_name << "[";
                first = true;
                for (const auto& idx : bc->m_rhs->m_indices)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "])";
                for (const auto& constraint : bc->m_constraints)
                {
                    s << ", " << constraint;
                }
                s << ";\n";
                if (bc->m_default.length())
                {
                    s << " default " << bc->m_default;
                }
                continue;
            }
        }
        {
            const UnaryContraction* uc = dynamic_cast<const UnaryContraction*>(stmt.get());
            if (uc)
            {
                if (!uc->m_output || !uc->m_input)
                {
                    throw std::logic_error{"Incomplete contraction"};
                }
                if (uc->m_output->m_indices.size() != uc->m_output->m_dims.size())
                {
                    throw std::logic_error{"Contraction index count != dimension count"};
                }
                s << uc->m_output->m_name << "[";
                first = true;
                for (const auto& idx : uc->m_output->m_indices)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                if (uc->m_output->m_indices.size())
                {
                    s << " : ";
                }
                first = true;
                for (const auto& dim : uc->m_output->m_dims)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << dim;
                }
                s << "] = " << uc->m_agg_op << "(" << uc->m_input->m_name << "[";
                first = true;
                for (const auto& idx : uc->m_input->m_indices)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "])";
                for (const auto& constraint : uc->m_constraints)
                {
                    s << ", " << constraint;
                }
                if (uc->m_default.length())
                {
                    s << " default " << uc->m_default;
                }
                s << ";\n";
                continue;
            }
        }
        {
            const Elementwise* e = dynamic_cast<const Elementwise*>(stmt.get());
            if (e)
            {
                s << e->m_lhs << " = " << e->m_rhs << ";\n";
                continue;
            }
        }
        throw std::logic_error{"Failed to determine dynamic operation class"};
    }
    s << "}";
    return s.str();
}

vp::application ngraph::runtime::plaidml::builder::Function::finalize() const
{
    std::vector<vp::variable> params;
    for (auto& input : m_inputs)
    {
        params.emplace_back(input.m_var);
    }
    auto str = to_string();
    if (m_debug)
    {
        PLAIDML_DEBUG << "Built Tile code:\n" << str;
    }
    return vp::function{str}.apply(params);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(Input input) &
{
    m_inputs.emplace_back(std::move(input));
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(Input input) &&
{
    m_inputs.emplace_back(std::move(input));
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(Output output) &
{
    m_outputs.emplace_back(std::move(output));
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(Output output) &&
{
    m_outputs.emplace_back(std::move(output));
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(UnaryContraction contraction) &
{
    m_stmts.emplace_back(
        std::unique_ptr<UnaryContraction>{new UnaryContraction(std::move(contraction))});
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(UnaryContraction contraction) &&
{
    m_stmts.emplace_back(
        std::unique_ptr<UnaryContraction>{new UnaryContraction(std::move(contraction))});
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(BinaryContraction contraction) &
{
    m_stmts.emplace_back(
        std::unique_ptr<BinaryContraction>{new BinaryContraction(std::move(contraction))});
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(BinaryContraction contraction) &&
{
    m_stmts.emplace_back(
        std::unique_ptr<BinaryContraction>{new BinaryContraction(std::move(contraction))});
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(TernaryContraction contraction) &
{
    m_stmts.emplace_back(
        std::unique_ptr<TernaryContraction>{new TernaryContraction(std::move(contraction))});
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(TernaryContraction contraction) &&
{
    m_stmts.emplace_back(
        std::unique_ptr<TernaryContraction>{new TernaryContraction(std::move(contraction))});
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(Elementwise elementwise) &
{
    m_stmts.emplace_back(std::unique_ptr<Elementwise>{new Elementwise(std::move(elementwise))});
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(Elementwise elementwise) &&
{
    m_stmts.emplace_back(std::unique_ptr<Elementwise>{new Elementwise(std::move(elementwise))});
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Input::Input(vp::variable var, std::string name)
    : m_var{std::move(var)}
    , m_name{std::move(name)}
{
}

ngraph::runtime::plaidml::builder::Input& ngraph::runtime::plaidml::builder::Input::add_dims(
    std::string prefix, std::size_t first, std::size_t limit) &
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        m_dims.emplace_back(prefix + std::to_string(idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::Input&& ngraph::runtime::plaidml::builder::Input::add_dims(
    std::string prefix, std::size_t first, std::size_t limit) &&
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        m_dims.emplace_back(prefix + std::to_string(idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Input& ngraph::runtime::plaidml::builder::Input::add_rdims(
    std::string prefix, std::size_t limit, std::size_t first) &
{
    for (std::size_t idx = limit; first < idx;)
    {
        m_dims.emplace_back(prefix + std::to_string(--idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::Input&& ngraph::runtime::plaidml::builder::Input::add_rdims(
    std::string prefix, std::size_t limit, std::size_t first) &&
{
    for (std::size_t idx = limit; first < idx;)
    {
        m_dims.emplace_back(prefix + std::to_string(--idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Input&
    ngraph::runtime::plaidml::builder::Input::add_dims(std::initializer_list<std::string> s) &
{
    m_dims.insert(m_dims.end(), s.begin(), s.end());
    return *this;
}

ngraph::runtime::plaidml::builder::Input&&
    ngraph::runtime::plaidml::builder::Input::add_dims(std::initializer_list<std::string> s) &&
{
    m_dims.insert(m_dims.end(), s.begin(), s.end());
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Output::Output(std::string name)
    : m_name{std::move(name)}
{
}

ngraph::runtime::plaidml::builder::Elementwise::Elementwise(std::string lhs, std::string rhs)
    : m_lhs{std::move(lhs)}
    , m_rhs{std::move(rhs)}
{
}

ngraph::runtime::plaidml::builder::ContractionOutput::ContractionOutput(std::string name)
    : m_name{std::move(name)}
{
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_indices(std::string prefix,
                                                                      std::size_t first,
                                                                      std::size_t limit) &
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        m_indices.emplace_back(prefix + std::to_string(idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionOutput&&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_indices(std::string prefix,
                                                                      std::size_t first,
                                                                      std::size_t limit) &&
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        m_indices.emplace_back(prefix + std::to_string(idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_rindices(std::string prefix,
                                                                       std::size_t limit,
                                                                       std::size_t first) &
{
    for (std::size_t idx = limit; first < idx;)
    {
        m_indices.emplace_back(prefix + std::to_string(--idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionOutput&&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_rindices(std::string prefix,
                                                                       std::size_t limit,
                                                                       std::size_t first) &&
{
    for (std::size_t idx = limit; first < idx;)
    {
        m_indices.emplace_back(prefix + std::to_string(--idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_indices(
        std::initializer_list<std::string> s) &
{
    m_indices.insert(m_indices.end(), s.begin(), s.end());
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionOutput&&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_indices(
        std::initializer_list<std::string> s) &&
{
    m_indices.insert(m_indices.end(), s.begin(), s.end());
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_dims(std::string prefix,
                                                                   std::size_t first,
                                                                   std::size_t limit) &
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        m_dims.emplace_back(prefix + std::to_string(idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionOutput&&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_dims(std::string prefix,
                                                                   std::size_t first,
                                                                   std::size_t limit) &&
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        m_dims.emplace_back(prefix + std::to_string(idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_rdims(std::string prefix,
                                                                    std::size_t limit,
                                                                    std::size_t first) &
{
    for (std::size_t idx = limit; first < idx;)
    {
        m_dims.emplace_back(prefix + std::to_string(--idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionOutput&&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_rdims(std::string prefix,
                                                                    std::size_t limit,
                                                                    std::size_t first) &&
{
    for (std::size_t idx = limit; first < idx;)
    {
        m_dims.emplace_back(prefix + std::to_string(--idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_dims(
        std::initializer_list<std::string> s) &
{
    m_dims.insert(m_dims.end(), s.begin(), s.end());
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionOutput&&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_dims(
        std::initializer_list<std::string> s) &&
{
    m_dims.insert(m_dims.end(), s.begin(), s.end());
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionInput&
    ngraph::runtime::plaidml::builder::ContractionInput::add_indices(std::string prefix,
                                                                     std::size_t first,
                                                                     std::size_t limit) &
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        m_indices.emplace_back(prefix + std::to_string(idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionInput&&
    ngraph::runtime::plaidml::builder::ContractionInput::add_indices(std::string prefix,
                                                                     std::size_t first,
                                                                     std::size_t limit) &&
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        m_indices.emplace_back(prefix + std::to_string(idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionInput&
    ngraph::runtime::plaidml::builder::ContractionInput::add_rindices(std::string prefix,
                                                                      std::size_t limit,
                                                                      std::size_t first) &
{
    for (std::size_t idx = limit; first < idx;)
    {
        m_indices.emplace_back(prefix + std::to_string(--idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionInput&&
    ngraph::runtime::plaidml::builder::ContractionInput::add_rindices(std::string prefix,
                                                                      std::size_t limit,
                                                                      std::size_t first) &&
{
    for (std::size_t idx = limit; first < idx;)
    {
        m_indices.emplace_back(prefix + std::to_string(--idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionInput&
    ngraph::runtime::plaidml::builder::ContractionInput::add_indices(
        std::initializer_list<std::string> s) &
{
    m_indices.insert(m_indices.end(), s.begin(), s.end());
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionInput&&
    ngraph::runtime::plaidml::builder::ContractionInput::add_indices(
        std::initializer_list<std::string> s) &&
{
    m_indices.insert(m_indices.end(), s.begin(), s.end());
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::UnaryContraction::UnaryContraction(std::string agg_op)
    : m_agg_op{std::move(agg_op)}
{
}

ngraph::runtime::plaidml::builder::UnaryContraction&
    ngraph::runtime::plaidml::builder::UnaryContraction::set(ContractionInput input) &
{
    m_input = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::UnaryContraction&&
    ngraph::runtime::plaidml::builder::UnaryContraction::set(ContractionInput input) &&
{
    m_input = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::UnaryContraction&
    ngraph::runtime::plaidml::builder::UnaryContraction::set(ContractionOutput output) &
{
    m_output = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return *this;
}

ngraph::runtime::plaidml::builder::UnaryContraction&&
    ngraph::runtime::plaidml::builder::UnaryContraction::set(ContractionOutput output) &&
{
    m_output = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::UnaryContraction&
    ngraph::runtime::plaidml::builder::UnaryContraction::set_default(std::string tensor) &
{
    m_default = std::move(tensor);
    return *this;
}

ngraph::runtime::plaidml::builder::UnaryContraction&&
    ngraph::runtime::plaidml::builder::UnaryContraction::set_default(std::string tensor) &&
{
    m_default = std::move(tensor);
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::BinaryContraction::BinaryContraction(std::string agg_op,
                                                                        std::string comb_op)
    : m_agg_op{std::move(agg_op)}
    , m_comb_op{std::move(comb_op)}
{
}

ngraph::runtime::plaidml::builder::BinaryContraction&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_lhs(ContractionInput input) &
{
    m_lhs = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::BinaryContraction&&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_lhs(ContractionInput input) &&
{
    m_lhs = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::BinaryContraction&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_rhs(ContractionInput input) &
{
    m_rhs = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::BinaryContraction&&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_rhs(ContractionInput input) &&
{
    m_rhs = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::BinaryContraction&
    ngraph::runtime::plaidml::builder::BinaryContraction::set(ContractionOutput output) &
{
    m_output = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return *this;
}

ngraph::runtime::plaidml::builder::BinaryContraction&&
    ngraph::runtime::plaidml::builder::BinaryContraction::set(ContractionOutput output) &&
{
    m_output = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::BinaryContraction&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_default(std::string tensor) &
{
    m_default = std::move(tensor);
    return *this;
}

ngraph::runtime::plaidml::builder::BinaryContraction&&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_default(std::string tensor) &&
{
    m_default = std::move(tensor);
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::TernaryContraction::TernaryContraction(std::string agg_op,
                                                                          std::string comb_op)
    : m_agg_op{std::move(agg_op)}
    , m_comb_op{std::move(comb_op)}
{
}

ngraph::runtime::plaidml::builder::TernaryContraction&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_first(ContractionInput input) &
{
    m_first = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::TernaryContraction&&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_first(ContractionInput input) &&
{
    m_first = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::TernaryContraction&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_second(ContractionInput input) &
{
    m_second = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::TernaryContraction&&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_second(ContractionInput input) &&
{
    m_second = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::TernaryContraction&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_third(ContractionInput input) &
{
    m_third = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::TernaryContraction&&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_third(ContractionInput input) &&
{
    m_third = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::TernaryContraction&
    ngraph::runtime::plaidml::builder::TernaryContraction::set(ContractionOutput output) &
{
    m_output = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return *this;
}

ngraph::runtime::plaidml::builder::TernaryContraction&&
    ngraph::runtime::plaidml::builder::TernaryContraction::set(ContractionOutput output) &&
{
    m_output = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return std::move(*this);
}
