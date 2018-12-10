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

#include <memory>
#include <unordered_map>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/runtime/gpu/gpu_op_annotations.hpp"
#include "ngraph/runtime/gpu/op/batch_norm.hpp"
#include "ngraph/runtime/gpu/pass/gpu_batch_norm_cache.hpp"

using namespace ngraph;

bool ngraph::runtime::gpu::pass::BatchNormCache::run_on_function(
    std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (auto bnbp = std::dynamic_pointer_cast<op::BatchNormTrainingBackprop>(n))
        {
            // batch norm bprop annotations are used to indicate if variance is in inverse stddev format
            auto op_annotations =
                std::make_shared<ngraph::runtime::gpu::BatchNormBackpropAnnotations>();

            // pass must be run prior to GOE elimination
            // collect all batch norm inputs to batch norm backward op
            std::vector<std::shared_ptr<op::GetOutputElement>> goes;
            for (auto& arg : bnbp->get_arguments())
            {
                if (auto goe = std::dynamic_pointer_cast<op::GetOutputElement>(arg))
                {
                    if (auto bn = std::dynamic_pointer_cast<op::BatchNormTraining>(
                            goe->get_arguments().at(0)))
                    {
                        goes.push_back(goe);
                    }
                }
            }

            // only replace if some of the inputs to backprop are from fprop directly
            if (goes.size())
            {
                if (auto target = std::dynamic_pointer_cast<op::BatchNormTraining>(
                        goes.front()->get_arguments().at(0)))
                {
                    auto replacement = std::make_shared<op::gpu::BatchNormTrainingWithStats>(
                        target->get_eps_value(),
                        target->get_argument(0),
                        target->get_argument(1),
                        target->get_argument(2));

                    // replace all users of old batchnorm with cudnn batchnorm
                    for (size_t i = 0; i < target->get_outputs().size(); i++)
                    {
                        auto& target_output = target->get_outputs().at(i);
                        std::set<ngraph::descriptor::Input*> copy_inputs{
                            begin(target_output.get_inputs()), end(target_output.get_inputs())};
                        for (auto input : copy_inputs)
                        {
                            input->replace_output(replacement->get_outputs().at(i));
                        }
                    }

                    // for each output of forward op into backprop op
                    // use the mean and inverse variance from the forward
                    // cudnn op to avoid recalculation of batch statistics
                    for (auto& goe : goes)
                    {
                        auto out_idx = goe->get_n();
                        if (out_idx != 0)
                        {
                            auto new_goe =
                                std::make_shared<op::GetOutputElement>(replacement, out_idx + 2);
                            ngraph::replace_node(goe, new_goe);
                        }
                    }
                    replaced = true;
                    op_annotations->set_inverted_variance(true);
                }
            }
            bnbp->set_op_annotations(op_annotations);
        }
    }
    return replaced;
}
