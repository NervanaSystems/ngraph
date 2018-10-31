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

#include "ngraph/log.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            // BatchNormInference implements batch normalization for inference, in
            // which the mean and variance to use are supplied.
            template <>
            void Impl<op::BatchNormInference>::operator()()
            {
                auto& input_shape = op().get_input_shape(2);
                check_inputs(5);
                check_outputs(1);

                auto f = start_tile_function();
                f.add(builder::Input{op_input(0), "Gamma"}.add_dims({"C"}))
                    .add(builder::Input{op_input(1), "Beta"}.add_dims({"C"}))
                    .add(builder::Input{op_input(2), "Input"}
                             .add_dims({"B", "C"})
                             .add_dims("DI", 3, input_shape.size() + 1))
                    .add(builder::Output{"Normalized"})
                    .add(builder::Input{op_input(3), "Mean"}.add_dims({"C"}))
                    .add(builder::Input{op_input(4), "Variance"}.add_dims({"C"}));

                std::string ones;
                for (auto idx = 2; idx < input_shape.size(); ++idx)
                {
                    ones += ", 1";
                }

                if (input_shape.size() <= 2)
                {
                    f.add(builder::Elementwise{"GammaP", "Gamma"})
                        .add(builder::Elementwise{"BetaP", "Beta"});
                }
                else
                {
                    f.add(builder::Elementwise{"GammaP",
                                               std::string{"reshape(Gamma, C"} + ones + ")"})
                        .add(builder::Elementwise{"BetaP",
                                                  std::string{"reshape(Beta, C"} + ones + ")"});
                }

                if (input_shape.size() <= 2)
                {
                    f.add(builder::Elementwise{"MeanP", "Mean"});
                }
                else
                {
                    f.add(
                        builder::Elementwise{"MeanP", std::string{"reshape(Mean, C"} + ones + ")"});
                }

                if (input_shape.size() <= 2)
                {
                    f.add(builder::Elementwise{"VarianceP", "Variance"});
                }
                else
                {
                    f.add(builder::Elementwise{"VarianceP",
                                               std::string{"reshape(Variance, C"} + ones + ")"});
                }

                f.add(builder::Elementwise{"Normalized",
                                           "(((Input-MeanP) / sqrt(VarianceP + " +
                                               std::to_string(op().get_eps_value()) +
                                               ")) * GammaP) + BetaP"});

                auto app = f.finalize();

                set_output(app);
            }

            // BatchNormTraining implements batch normalization for training, in
            // which the mean and variance are to be computed from the supplied
            // input.
            template <>
            void Impl<op::BatchNormTraining>::operator()()
            {
                auto& input_shape = op().get_input_shape(2);
                check_inputs(3);
                check_outputs(3);

                auto f = start_tile_function();
                f.add(builder::Input{op_input(0), "Gamma"}.add_dims({"C"}))
                    .add(builder::Input{op_input(1), "Beta"}.add_dims({"C"}))
                    .add(builder::Input{op_input(2), "Input"}
                             .add_dims({"B", "C"})
                             .add_dims("DI", 3, input_shape.size() + 1))
                    .add(builder::Output{"Normalized"})
                    .add(builder::Output{"Mean"})
                    .add(builder::Output{"Variance"});

                std::string ones;
                for (auto idx = 2; idx < input_shape.size(); ++idx)
                {
                    ones += ", 1";
                }

                if (input_shape.size() <= 2)
                {
                    f.add(builder::Elementwise{"GammaP", "Gamma"})
                        .add(builder::Elementwise{"BetaP", "Beta"});
                }
                else
                {
                    f.add(builder::Elementwise{"GammaP",
                                               std::string{"reshape(Gamma, C"} + ones + ")"})
                        .add(builder::Elementwise{"BetaP",
                                                  std::string{"reshape(Beta, C"} + ones + ")"});
                }

                if (input_shape.size() <= 2)
                {
                    f.add(builder::Elementwise{"EltCount", "B"});
                }
                else
                {
                    std::string elts{"B"};
                    for (auto idx = 2; idx < input_shape.size(); ++idx)
                    {
                        elts += " * DI" + std::to_string(idx + 1);
                    }
                    f.add(builder::Elementwise{"EltCount", std::move(elts)});
                }

                f.add(builder::UnaryContraction{"+"}
                          .set(builder::ContractionOutput{"SumInput"}.add_indices({"c"}).add_dims(
                              {"C"}))
                          .set(builder::ContractionInput{"Input"}
                                   .add_indices({"b", "c"})
                                   .add_indices("di", 3, input_shape.size() + 1)));
                f.add(builder::Elementwise{"Mean", "SumInput / EltCount"});

                if (input_shape.size() <= 2)
                {
                    f.add(builder::Elementwise{"MeanP", "Mean"});
                }
                else
                {
                    f.add(
                        builder::Elementwise{"MeanP", std::string{"reshape(Mean, C"} + ones + ")"});
                }

                f.add(builder::Elementwise{"DiffV", "(Input - MeanP)"})
                    .add(builder::Elementwise{"SqDiffV", "DiffV*DiffV"})
                    .add(builder::UnaryContraction{"+"}
                             .set(builder::ContractionOutput{"SumSqDiffV"}
                                      .add_indices({"c"})
                                      .add_dims({"C"}))
                             .set(builder::ContractionInput{"SqDiffV"}
                                      .add_indices({"b", "c"})
                                      .add_indices("di", 3, input_shape.size() + 1)))
                    .add(builder::Elementwise{"Variance", "SumSqDiffV / EltCount"});

                if (input_shape.size() <= 2)
                {
                    f.add(builder::Elementwise{"VarianceP", "Variance"});
                }
                else
                {
                    f.add(builder::Elementwise{"VarianceP",
                                               std::string{"reshape(Variance, C"} + ones + ")"});
                }

                f.add(builder::Elementwise{"Normalized",
                                           "(((Input-MeanP) / sqrt(VarianceP + " +
                                               std::to_string(op().get_eps_value()) +
                                               ")) * GammaP) + BetaP"});

                auto app = f.finalize();

                set_output(0, app.get_output(0));
                set_output(1, app.get_output(1));
                set_output(2, app.get_output(2));
            }

            template <>
            void Impl<op::BatchNormTrainingBackprop>::operator()()
            {
                // WARNING: I'm unconvinced that we have sufficient test converage for BatchNorm
                // backprop and in particular I'm concerned that Gamma/Beta and Mean/Var could be
                // swapped without the tests catching it.
                check_inputs(6);
                check_outputs(3);
                auto& input_shape = op().get_input_shape(2);
                std::string epsilon = std::to_string(op().get_eps_value());

                auto f = start_tile_function();
                // Header
                f.add(builder::Input{op_input(0), "Gamma"}.add_dims({"C"}))
                    .add(builder::Input{op_input(1), "Beta"}.add_dims({"C"}))
                    .add(builder::Input{op_input(2), "Input"}
                             .add_dims({"N", "C"})
                             .add_dims("X", 3, input_shape.size() + 1))
                    .add(builder::Input{op_input(3), "Mean"}.add_dims({"C"}))
                    .add(builder::Input{op_input(4), "Var"}.add_dims({"C"}))
                    .add(builder::Input{op_input(5), "DOutput"}
                             .add_dims({"N", "C"})
                             .add_dims("X", 3, input_shape.size() + 1));
                f.add(builder::Output{"DInput"});
                f.add(builder::Output{"DGamma"});
                f.add(builder::Output{"DBeta"});

                // Prep for body
                builder::ContractionOutput broadcast_gamma{"BroadcastGamma"};
                builder::ContractionOutput broadcast_dgamma{"BroadcastDGamma"};
                builder::ContractionOutput broadcast_dbeta{"BroadcastDBeta"};
                broadcast_gamma.add_indices({"0", "c"}).add_dims({"1", "C"});
                broadcast_dgamma.add_indices({"0", "c"}).add_dims({"1", "C"});
                broadcast_dbeta.add_indices({"0", "c"}).add_dims({"1", "C"});
                for (std::size_t i = 0; i < input_shape.size() - 2; ++i)
                {
                    broadcast_gamma.add_indices({"0"}).add_dims({"1"});
                    broadcast_dgamma.add_indices({"0"}).add_dims({"1"});
                    broadcast_dbeta.add_indices({"0"}).add_dims({"1"});
                }
                std::ostringstream reduction_dims;
                reduction_dims << "("
                               << "N";
                for (std::size_t i = 3; i < input_shape.size() + 1; ++i)
                {
                    reduction_dims << " * X" << i;
                }
                reduction_dims << ")";

                // Body
                f.add(builder::UnaryContraction{"+"}
                          .set(builder::ContractionOutput{"BatchMeanNumerator"}
                                   .add_indices({"0", "c", "0", "0"})
                                   .add_dims({"1", "C", "1", "1"}))
                          .set(builder::ContractionInput{"Input"}
                                   .add_indices({"n", "c"})
                                   .add_indices("x", 3, input_shape.size() + 1)));
                f.add(builder::Elementwise{"BatchMean",
                                           "BatchMeanNumerator / " + reduction_dims.str()});
                f.add(builder::Elementwise{"NegBatchMean", "-BatchMean"});
                f.add(builder::BinaryContraction{"=", "+"}
                          .set(builder::ContractionOutput{"Deviation"}
                                   .add_indices({"n", "c"})
                                   .add_indices("x", 3, input_shape.size() + 1)
                                   .add_dims({"N", "C"})
                                   .add_dims("X", 3, input_shape.size() + 1))
                          .set_lhs(builder::ContractionInput{"Input"}
                                       .add_indices({"n", "c"})
                                       .add_indices("x", 3, input_shape.size() + 1))
                          .set_rhs(builder::ContractionInput{"NegBatchMean"}.add_indices(
                              {"0", "c", "0", "0"})));
                f.add(builder::BinaryContraction{"+", "*"}
                          .set(builder::ContractionOutput{"BatchVarNumerator"}
                                   .add_indices({"0", "c", "0", "0"})
                                   .add_dims({"1", "C", "1", "1"}))
                          .set_lhs(builder::ContractionInput{"Deviation"}
                                       .add_indices({"n", "c"})
                                       .add_indices("x", 3, input_shape.size() + 1))
                          .set_rhs(builder::ContractionInput{"Deviation"}
                                       .add_indices({"n", "c"})
                                       .add_indices("x", 3, input_shape.size() + 1)));
                f.add(builder::Elementwise{"BatchVar",
                                           "BatchVarNumerator / " + reduction_dims.str()});
                f.add(builder::Elementwise{"BatchStdDev", "sqrt(BatchVar + " + epsilon + ")"});
                f.add(builder::Elementwise{"NormedInput", "(Input - BatchMean) / BatchStdDev"});

                f.add(builder::Elementwise{"ZeroedInput", "Input - BatchMean"});
                f.add(builder::UnaryContraction{"="}
                          .set(broadcast_gamma)
                          .set(builder::ContractionInput{"Gamma"}.add_indices({"c"})));
                f.add(builder::Elementwise{"DNormedInput", "DOutput * BroadcastGamma"});

                f.add(builder::UnaryContraction{"+"}
                          .set(builder::ContractionOutput{"SumDOutput"}.add_indices({"c"}).add_dims(
                              {"C"}))
                          .set(builder::ContractionInput{"DOutput"}
                                   .add_indices({"n", "c"})
                                   .add_indices("x", 3, input_shape.size() + 1)));
                f.add(builder::BinaryContraction{"+", "*"}
                          .set(builder::ContractionOutput{"DGamma"}.add_indices({"c"}).add_dims(
                              {"C"}))
                          .set_lhs(builder::ContractionInput{"DOutput"}
                                       .add_indices({"n", "c"})
                                       .add_indices("x", 3, input_shape.size() + 1))
                          .set_rhs(builder::ContractionInput{"NormedInput"}
                                       .add_indices({"n", "c"})
                                       .add_indices("x", 3, input_shape.size() + 1)));
                f.add(builder::Elementwise{"DBeta", "SumDOutput"});
                f.add(builder::UnaryContraction{"="}
                          .set(broadcast_dgamma)
                          .set(builder::ContractionInput{"DGamma"}.add_indices({"c"})));
                f.add(builder::UnaryContraction{"="}
                          .set(broadcast_dbeta)
                          .set(builder::ContractionInput{"DBeta"}.add_indices({"c"})));
                f.add(builder::Elementwise{"DInput",
                                           "(BroadcastGamma / BatchStdDev) * (DOutput - "
                                           "(NormedInput * BroadcastDGamma + BroadcastDBeta) / (" +
                                               reduction_dims.str() + "))"});

                // Return results
                auto app = f.finalize();
                set_output(0, app.get_output(0));
                set_output(1, app.get_output(1));
                set_output(2, app.get_output(2));
            }

            namespace
            {
                Impl<op::BatchNormInference>::Registration register_batch_norm_inference;
                Impl<op::BatchNormTraining>::Registration register_batch_norm_training;
                Impl<op::BatchNormTrainingBackprop>::Registration
                    register_batch_norm_training_backprop;
            }
        }
    }
}
