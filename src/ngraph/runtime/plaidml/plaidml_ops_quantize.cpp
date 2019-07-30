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

#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplDequantize, OpImpl<op::Dequantize>);
            NGRAPH_PLAIDML_OP_CLASS(ImplQuantize, OpImpl<op::Quantize>);
        }
    }
}

void ngraph::runtime::plaidml::ImplDequantize::Apply()
{
    check_inputs(3);
    check_outputs(1);

    const auto& axes = op().get_axes();

    const auto& input_shape = op().get_input_shape(0);
    const auto& scale_shape = op().get_input_shape(1);
    const auto& zp_shape = op().get_input_shape(2);

    const auto& input_type = op().get_input_element_type(0);

    if (!input_type.is_signed() && input_type.size() >= 8)
    {
        throw std::runtime_error("PlaidML does not yet support dequantizing from uint64+");
    }

    if (scale_shape != zp_shape)
    {
        throw std::runtime_error("Dequantize given mismatched scale & zero point shapes.");
    }

    if (scale_shape.size() != axes.size())
    {
        std::ostringstream msg;
        msg << "Dequantize received " << axes.size()
            << " axes to use for scale & zero point, but those tensors have " << scale_shape.size()
            << " dimensions instead.";
        throw std::runtime_error(msg.str());
    }

    std::vector<std::string> short_idxs;
    for (size_t i = 0; i < input_shape.size(); ++i)
    {
        if (axes.count(i))
        {
            std::ostringstream name;
            name << "i" << i;
            short_idxs.push_back(name.str());
        }
    }

    builder::ContractionInput scale_input{"S"};
    builder::ContractionInput neg_zp_input{"NegZ"};
    for (const auto& idx : short_idxs)
    {
        scale_input.add_indices({idx});
        neg_zp_input.add_indices({idx});
    }

    std::function<std::string(std::string)> cast_uint_to_wider_int =
        [input_type](std::string tensor_name) {
            std::ostringstream cast_str;
            if (!input_type.is_signed())
            {
                cast_str << "as_int(" << tensor_name << ", " << 2 * 8 * input_type.size() << ")";
            }
            else
            {
                cast_str << tensor_name;
            }
            return cast_str.str();
        };
    builder::Elementwise CastI{"CastI", cast_uint_to_wider_int("I")};
    builder::Elementwise CastZ{"CastZ", cast_uint_to_wider_int("Z")};

    auto f = start_tile_function();
    f.add(builder::Input{op_input(0), "I"}.add_dims("I", 0, input_shape.size()))
        .add(builder::Input{op_input(1), "S"}.add_dims("S", 0, scale_shape.size()))
        .add(builder::Input{op_input(2), "Z"}.add_dims("Z", 0, zp_shape.size()))
        .add(builder::Output{"O"})
        .add(CastI)
        .add(CastZ)
        .add(builder::Elementwise{"NegZ", "-CastZ"})
        .add(
            builder::BinaryContraction{"=", "+"}
                .set(builder::ContractionOutput{"Offset"}
                         .add_indices("i", 0, input_shape.size())
                         .add_dims("I", 0, input_shape.size()))
                .set_lhs(builder::ContractionInput{"CastI"}.add_indices("i", 0, input_shape.size()))
                .set_rhs(neg_zp_input))
        .add(builder::BinaryContraction{"=", "*"}
                 .set(builder::ContractionOutput{"O"}
                          .add_indices("i", 0, input_shape.size())
                          .add_dims("I", 0, input_shape.size()))
                 .set_lhs(
                     builder::ContractionInput{"Offset"}.add_indices("i", 0, input_shape.size()))
                 .set_rhs(scale_input));

    set_output(f.finalize());
}

void ngraph::runtime::plaidml::ImplQuantize::Apply()
{
    check_inputs(3);
    check_outputs(1);

    const auto& type = op().get_output_element_type(0);
    const auto& axes = op().get_axes();
    const auto& round_mode = op().get_round_mode();

    const auto& input_shape = op().get_input_shape(0);
    const auto& scale_shape = op().get_input_shape(1);
    const auto& zp_shape = op().get_input_shape(2);

    std::function<std::string(std::string)> cast_to_output_type = [type](std::string tensor_name) {
        std::ostringstream cast_str;
        if (type.is_signed())
        {
            cast_str << "as_int";
        }
        else
        {
            cast_str << "as_uint";
        }
        cast_str << "(" << tensor_name << ", " << 8 * type.size() << ")";
        return cast_str.str();
    };

    if (scale_shape != zp_shape)
    {
        throw std::runtime_error("Quantize given mismatched scale & zero point shapes.");
    }

    if (scale_shape.size() != axes.size())
    {
        std::ostringstream msg;
        msg << "Quantize received " << axes.size()
            << " axes to use for scale & zero point, but those tensors have " << scale_shape.size()
            << " dimensions instead.";
        throw std::runtime_error(msg.str());
    }

    std::vector<std::string> short_idxs;
    for (size_t i = 0; i < input_shape.size(); ++i)
    {
        if (axes.count(i))
        {
            std::ostringstream name;
            name << "i" << i;
            short_idxs.push_back(name.str());
        }
    }

    if (!type.is_integral())
    {
        throw std::runtime_error("Quantize output type must be integral");
    }

    builder::Elementwise Rounded{"Rounded", ""};
    builder::Elementwise Clamped{"Clamped", ""};
    builder::Elementwise O{"O", ""};

    int64_t q_min;
    int64_t q_max;
    std::ostringstream clamp_formula;
    if (type.size() > 4)
    {
        // PlaidML doesn't support quantization clamping for types wider than 32 bits
        if (!type.is_signed())
        {
            clamp_formula << "Uncast < 0 ? 0 : Uncast";
        }
        else
        {
            clamp_formula << "Uncast";
        }
    }
    else
    {
        if (type.is_signed())
        {
            q_max = (1 << (8 * type.size() - 1)) - 1;
            q_min = -q_max - 1;
        }
        else
        {
            q_max = (1 << (8 * type.size())) - 1;
            q_min = 0;
        }
        clamp_formula << "Uncast < " << q_min << " ? " << q_min << " : "
                      << "(Uncast > " << q_max << " ? " << q_max << " : Uncast)";
    }
    Clamped.set_rhs(clamp_formula.str());

    std::ostringstream round_formula;
    std::string lower_rounded_int;
    switch (round_mode)
    {
    case ngraph::op::Quantize::RoundMode::ROUND_DOWN: Rounded.set_rhs("floor(Frac)"); break;
    case ngraph::op::Quantize::RoundMode::ROUND_UP: Rounded.set_rhs("ceil(Frac)"); break;
    case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_DOWNWARD:
        Rounded.set_rhs("ceil(Frac - 0.5)");
        break;
    case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_UPWARD:
        Rounded.set_rhs("floor(Frac + 0.5)");
        break;
    case ngraph::op::Quantize::RoundMode::ROUND_TOWARD_ZERO:
        Rounded.set_rhs("Frac > 0 ? floor(Frac) : ceil(Frac)");
        break;
    case ngraph::op::Quantize::RoundMode::ROUND_TOWARD_INFINITY:
        Rounded.set_rhs("Frac < 0 ? floor(Frac) : ceil(Frac)");
        break;
    case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_ZERO:
        Rounded.set_rhs("Frac > 0 ? ceil(Frac - 0.5) : floor(Frac + 0.5)");
        break;
    case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY:
        Rounded.set_rhs("Frac < 0 ? ceil(Frac - 0.5) : floor(Frac + 0.5)");
        break;
    case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN:
        // This is ugly, but it produces correct output
        lower_rounded_int = cast_to_output_type("ceil(Frac - 0.5)");
        round_formula << "2 * (" << lower_rounded_int << " / 2) == " << lower_rounded_int
                      << " ? ceil(Frac - 0.5) : floor(Frac + 0.5)";
        Rounded.set_rhs(round_formula.str());
        break;
    default:
        throw std::runtime_error("Requested quantize round mode not yet implemented in PlaidML");
    }

    O.set_rhs(cast_to_output_type("Clamped"));

    builder::ContractionInput scale_recip_input{"SRecip"};
    builder::ContractionInput zp_input{"Z"};
    for (const auto& idx : short_idxs)
    {
        scale_recip_input.add_indices({idx});
        zp_input.add_indices({idx});
    }

    auto f = start_tile_function();
    f.add(builder::Input{op_input(0), "I"}.add_dims("I", 0, input_shape.size()))
        .add(builder::Input{op_input(1), "S"}.add_dims("S", 0, scale_shape.size()))
        .add(builder::Input{op_input(2), "Z"}.add_dims("Z", 0, zp_shape.size()))
        .add(builder::Output{"O"})
        .add(builder::Elementwise{"SRecip", "1 / S"})
        .add(builder::BinaryContraction{"=", "*"}
                 .set(builder::ContractionOutput{"Frac"}
                          .add_indices("i", 0, input_shape.size())
                          .add_dims("I", 0, input_shape.size()))
                 .set_lhs(builder::ContractionInput{"I"}.add_indices("i", 0, input_shape.size()))
                 .set_rhs(scale_recip_input))
        .add(Rounded)
        .add(builder::BinaryContraction{"=", "+"}
                 .set(builder::ContractionOutput{"Uncast"}
                          .add_indices("i", 0, input_shape.size())
                          .add_dims("I", 0, input_shape.size()))
                 .set_lhs(
                     builder::ContractionInput{"Rounded"}.add_indices("i", 0, input_shape.size()))
                 .set_rhs(zp_input))
        .add(Clamped)
        .add(O);

    set_output(f.finalize());
}
