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

#include "ngraph/runtime/plaidml/plaidml_builder.hpp"

#include "ngraph/runtime/plaidml/plaidml_logger.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace vp = vertexai::plaidml;

ngraph::runtime::plaidml::builder::Function::Function(const std::string& name, bool debug)
    : name_{name}
    , debug_{debug}
{
}

std::string ngraph::runtime::plaidml::builder::Function::to_string() const
{
    std::ostringstream s;
    s << "function (";
    bool first = true;
    for (const auto& input : inputs_)
    {
        if (!first)
        {
            s << ", ";
        }
        first = false;
        s << input.name_;
        if (input.dims_.size())
        {
            s << "[";
            bool first_dim = true;
            for (const auto& dim : input.dims_)
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
    for (const auto& output : outputs_)
    {
        if (!first)
        {
            s << ", ";
        }
        first = false;
        s << output.name_;
    }
    s << ") {\n";
    std::string name_annotation;
    if (name_.size())
    {
        name_annotation = "[[name(op" + name_ + ")]]\n  ";
    }
    for (const std::unique_ptr<Statement>& stmt : stmts_)
    {
        s << "  " << name_annotation;
        {
            const TernaryContraction* tc = dynamic_cast<const TernaryContraction*>(stmt.get());
            if (tc)
            {
                if (!tc->output_ || !tc->first_ || !tc->second_ || !tc->third_)
                {
                    throw std::logic_error{"Incomplete contraction"};
                }
                if (tc->output_->indices_.size() != tc->output_->dims_.size())
                {
                    throw std::logic_error{"Contraction index count != dimension count"};
                }
                s << tc->output_->name_ << "[";
                first = true;
                for (const auto& idx : tc->output_->indices_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                if (tc->output_->indices_.size())
                {
                    s << " : ";
                }
                first = true;
                for (const auto& dim : tc->output_->dims_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << dim;
                }
                s << "] = " << tc->agg_op_ << "(" << tc->first_->name_ << "[";
                first = true;
                for (const auto& idx : tc->first_->indices_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "] == " << tc->second_->name_ << "[";
                first = true;
                for (const auto& idx : tc->second_->indices_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "] " << tc->comb_op_ << " " << tc->third_->name_ << "[";
                first = true;
                for (const auto& idx : tc->third_->indices_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "])";
                for (const auto& constraint : tc->constraints_)
                {
                    s << ", " << constraint;
                }
                s << ";\n";
                continue;
            }
            const BinaryContraction* bc = dynamic_cast<const BinaryContraction*>(stmt.get());
            if (bc)
            {
                if (!bc->output_ || !bc->lhs_ || !bc->rhs_)
                {
                    throw std::logic_error{"Incomplete contraction"};
                }
                if (bc->output_->indices_.size() != bc->output_->dims_.size())
                {
                    throw std::logic_error{"Contraction index count != dimension count"};
                }
                s << bc->output_->name_ << "[";
                first = true;
                for (const auto& idx : bc->output_->indices_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                if (bc->output_->indices_.size())
                {
                    s << " : ";
                }
                first = true;
                for (const auto& dim : bc->output_->dims_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << dim;
                }
                s << "] = " << bc->agg_op_ << "(" << bc->lhs_->name_ << "[";
                first = true;
                for (const auto& idx : bc->lhs_->indices_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "] " << bc->comb_op_ << " " << bc->rhs_->name_ << "[";
                first = true;
                for (const auto& idx : bc->rhs_->indices_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "])";
                for (const auto& constraint : bc->constraints_)
                {
                    s << ", " << constraint;
                }
                s << ";\n";
                if (bc->default_.length())
                {
                    s << " default " << bc->default_;
                }
                continue;
            }
        }
        {
            const UnaryContraction* uc = dynamic_cast<const UnaryContraction*>(stmt.get());
            if (uc)
            {
                if (!uc->output_ || !uc->input_)
                {
                    throw std::logic_error{"Incomplete contraction"};
                }
                if (uc->output_->indices_.size() != uc->output_->dims_.size())
                {
                    throw std::logic_error{"Contraction index count != dimension count"};
                }
                s << uc->output_->name_ << "[";
                first = true;
                for (const auto& idx : uc->output_->indices_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                if (uc->output_->indices_.size())
                {
                    s << " : ";
                }
                first = true;
                for (const auto& dim : uc->output_->dims_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << dim;
                }
                s << "] = " << uc->agg_op_ << "(" << uc->input_->name_ << "[";
                first = true;
                for (const auto& idx : uc->input_->indices_)
                {
                    if (!first)
                    {
                        s << ", ";
                    }
                    first = false;
                    s << idx;
                }
                s << "])";
                for (const auto& constraint : uc->constraints_)
                {
                    s << ", " << constraint;
                }
                if (uc->default_.length())
                {
                    s << " default " << uc->default_;
                }
                s << ";\n";
                continue;
            }
        }
        {
            const Elementwise* e = dynamic_cast<const Elementwise*>(stmt.get());
            if (e)
            {
                s << e->lhs_ << " = " << e->rhs_ << ";\n";
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
    for (auto& input : inputs_)
    {
        params.emplace_back(input.var_);
    }
    auto str = to_string();
    if (debug_)
    {
        PLAIDML_DEBUG << "Built Tile code:\n" << str;
    }
    return vp::function{str}.apply(params);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(Input input) &
{
    inputs_.emplace_back(std::move(input));
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(Input input) &&
{
    inputs_.emplace_back(std::move(input));
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(Output output) &
{
    outputs_.emplace_back(std::move(output));
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(Output output) &&
{
    outputs_.emplace_back(std::move(output));
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(UnaryContraction contraction) &
{
    stmts_.emplace_back(
        std::unique_ptr<UnaryContraction>{new UnaryContraction(std::move(contraction))});
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(UnaryContraction contraction) &&
{
    stmts_.emplace_back(
        std::unique_ptr<UnaryContraction>{new UnaryContraction(std::move(contraction))});
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(BinaryContraction contraction) &
{
    stmts_.emplace_back(
        std::unique_ptr<BinaryContraction>{new BinaryContraction(std::move(contraction))});
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(BinaryContraction contraction) &&
{
    stmts_.emplace_back(
        std::unique_ptr<BinaryContraction>{new BinaryContraction(std::move(contraction))});
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(TernaryContraction contraction) &
{
    stmts_.emplace_back(
        std::unique_ptr<TernaryContraction>{new TernaryContraction(std::move(contraction))});
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(TernaryContraction contraction) &&
{
    stmts_.emplace_back(
        std::unique_ptr<TernaryContraction>{new TernaryContraction(std::move(contraction))});
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Function&
    ngraph::runtime::plaidml::builder::Function::add(Elementwise elementwise) &
{
    stmts_.emplace_back(std::unique_ptr<Elementwise>{new Elementwise(std::move(elementwise))});
    return *this;
}

ngraph::runtime::plaidml::builder::Function&&
    ngraph::runtime::plaidml::builder::Function::add(Elementwise elementwise) &&
{
    stmts_.emplace_back(std::unique_ptr<Elementwise>{new Elementwise(std::move(elementwise))});
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Input::Input(vp::variable var, std::string name)
    : var_{std::move(var)}
    , name_{std::move(name)}
{
}

ngraph::runtime::plaidml::builder::Input& ngraph::runtime::plaidml::builder::Input::add_dims(
    std::string prefix, std::size_t first, std::size_t limit) &
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        dims_.emplace_back(prefix + std::to_string(idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::Input&& ngraph::runtime::plaidml::builder::Input::add_dims(
    std::string prefix, std::size_t first, std::size_t limit) &&
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        dims_.emplace_back(prefix + std::to_string(idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Input& ngraph::runtime::plaidml::builder::Input::add_rdims(
    std::string prefix, std::size_t limit, std::size_t first) &
{
    for (std::size_t idx = limit; first < idx;)
    {
        dims_.emplace_back(prefix + std::to_string(--idx));
    }
    return *this;
}

ngraph::runtime::plaidml::builder::Input&& ngraph::runtime::plaidml::builder::Input::add_rdims(
    std::string prefix, std::size_t limit, std::size_t first) &&
{
    for (std::size_t idx = limit; first < idx;)
    {
        dims_.emplace_back(prefix + std::to_string(--idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Input&
    ngraph::runtime::plaidml::builder::Input::add_dims(std::initializer_list<std::string> s) &
{
    dims_.insert(dims_.end(), s.begin(), s.end());
    return *this;
}

ngraph::runtime::plaidml::builder::Input&&
    ngraph::runtime::plaidml::builder::Input::add_dims(std::initializer_list<std::string> s) &&
{
    dims_.insert(dims_.end(), s.begin(), s.end());
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::Output::Output(std::string name)
    : name_{std::move(name)}
{
}

ngraph::runtime::plaidml::builder::Elementwise::Elementwise(std::string lhs, std::string rhs)
    : lhs_{std::move(lhs)}
    , rhs_{std::move(rhs)}
{
}

ngraph::runtime::plaidml::builder::ContractionOutput::ContractionOutput(std::string name)
    : name_{std::move(name)}
{
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_indices(std::string prefix,
                                                                      std::size_t first,
                                                                      std::size_t limit) &
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        indices_.emplace_back(prefix + std::to_string(idx));
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
        indices_.emplace_back(prefix + std::to_string(idx));
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
        indices_.emplace_back(prefix + std::to_string(--idx));
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
        indices_.emplace_back(prefix + std::to_string(--idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_indices(
        std::initializer_list<std::string> s) &
{
    indices_.insert(indices_.end(), s.begin(), s.end());
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionOutput&&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_indices(
        std::initializer_list<std::string> s) &&
{
    indices_.insert(indices_.end(), s.begin(), s.end());
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_dims(std::string prefix,
                                                                   std::size_t first,
                                                                   std::size_t limit) &
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        dims_.emplace_back(prefix + std::to_string(idx));
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
        dims_.emplace_back(prefix + std::to_string(idx));
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
        dims_.emplace_back(prefix + std::to_string(--idx));
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
        dims_.emplace_back(prefix + std::to_string(--idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionOutput&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_dims(
        std::initializer_list<std::string> s) &
{
    dims_.insert(dims_.end(), s.begin(), s.end());
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionOutput&&
    ngraph::runtime::plaidml::builder::ContractionOutput::add_dims(
        std::initializer_list<std::string> s) &&
{
    dims_.insert(dims_.end(), s.begin(), s.end());
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionInput&
    ngraph::runtime::plaidml::builder::ContractionInput::add_indices(std::string prefix,
                                                                     std::size_t first,
                                                                     std::size_t limit) &
{
    for (std::size_t idx = first; idx < limit; ++idx)
    {
        indices_.emplace_back(prefix + std::to_string(idx));
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
        indices_.emplace_back(prefix + std::to_string(idx));
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
        indices_.emplace_back(prefix + std::to_string(--idx));
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
        indices_.emplace_back(prefix + std::to_string(--idx));
    }
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::ContractionInput&
    ngraph::runtime::plaidml::builder::ContractionInput::add_indices(
        std::initializer_list<std::string> s) &
{
    indices_.insert(indices_.end(), s.begin(), s.end());
    return *this;
}

ngraph::runtime::plaidml::builder::ContractionInput&&
    ngraph::runtime::plaidml::builder::ContractionInput::add_indices(
        std::initializer_list<std::string> s) &&
{
    indices_.insert(indices_.end(), s.begin(), s.end());
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::UnaryContraction::UnaryContraction(std::string agg_op)
    : agg_op_{std::move(agg_op)}
{
}

ngraph::runtime::plaidml::builder::UnaryContraction&
    ngraph::runtime::plaidml::builder::UnaryContraction::set(ContractionInput input) &
{
    input_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::UnaryContraction&&
    ngraph::runtime::plaidml::builder::UnaryContraction::set(ContractionInput input) &&
{
    input_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::UnaryContraction&
    ngraph::runtime::plaidml::builder::UnaryContraction::set(ContractionOutput output) &
{
    output_ = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return *this;
}

ngraph::runtime::plaidml::builder::UnaryContraction&&
    ngraph::runtime::plaidml::builder::UnaryContraction::set(ContractionOutput output) &&
{
    output_ = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::UnaryContraction&
    ngraph::runtime::plaidml::builder::UnaryContraction::set_default(std::string tensor) &
{
    default_ = std::move(tensor);
    return *this;
}

ngraph::runtime::plaidml::builder::UnaryContraction&&
    ngraph::runtime::plaidml::builder::UnaryContraction::set_default(std::string tensor) &&
{
    default_ = std::move(tensor);
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::BinaryContraction::BinaryContraction(std::string agg_op,
                                                                        std::string comb_op)
    : agg_op_{std::move(agg_op)}
    , comb_op_{std::move(comb_op)}
{
}

ngraph::runtime::plaidml::builder::BinaryContraction&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_lhs(ContractionInput input) &
{
    lhs_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::BinaryContraction&&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_lhs(ContractionInput input) &&
{
    lhs_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::BinaryContraction&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_rhs(ContractionInput input) &
{
    rhs_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::BinaryContraction&&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_rhs(ContractionInput input) &&
{
    rhs_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::BinaryContraction&
    ngraph::runtime::plaidml::builder::BinaryContraction::set(ContractionOutput output) &
{
    output_ = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return *this;
}

ngraph::runtime::plaidml::builder::BinaryContraction&&
    ngraph::runtime::plaidml::builder::BinaryContraction::set(ContractionOutput output) &&
{
    output_ = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::BinaryContraction&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_default(std::string tensor) &
{
    default_ = std::move(tensor);
    return *this;
}

ngraph::runtime::plaidml::builder::BinaryContraction&&
    ngraph::runtime::plaidml::builder::BinaryContraction::set_default(std::string tensor) &&
{
    default_ = std::move(tensor);
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::TernaryContraction::TernaryContraction(std::string agg_op,
                                                                          std::string comb_op)
    : agg_op_{std::move(agg_op)}
    , comb_op_{std::move(comb_op)}
{
}

ngraph::runtime::plaidml::builder::TernaryContraction&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_first(ContractionInput input) &
{
    first_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::TernaryContraction&&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_first(ContractionInput input) &&
{
    first_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::TernaryContraction&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_second(ContractionInput input) &
{
    second_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::TernaryContraction&&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_second(ContractionInput input) &&
{
    second_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::TernaryContraction&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_third(ContractionInput input) &
{
    third_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return *this;
}

ngraph::runtime::plaidml::builder::TernaryContraction&&
    ngraph::runtime::plaidml::builder::TernaryContraction::set_third(ContractionInput input) &&
{
    third_ = std::unique_ptr<ContractionInput>{new ContractionInput(std::move(input))};
    return std::move(*this);
}

ngraph::runtime::plaidml::builder::TernaryContraction&
    ngraph::runtime::plaidml::builder::TernaryContraction::set(ContractionOutput output) &
{
    output_ = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return *this;
}

ngraph::runtime::plaidml::builder::TernaryContraction&&
    ngraph::runtime::plaidml::builder::TernaryContraction::set(ContractionOutput output) &&
{
    output_ = std::unique_ptr<ContractionOutput>{new ContractionOutput(std::move(output))};
    return std::move(*this);
}
