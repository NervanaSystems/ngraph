//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "gpu_op_annotations.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "op/batch_norm.hpp"
#include "pass/gpu_batch_norm_cache.hpp"

using namespace ngraph;

bool ngraph::runtime::gpu::pass::BatchNormCache::run_on_function(
    std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (auto bnbp = std::dynamic_pointer_cast<op::v0::BatchNormTrainingBackprop>(n))
        {
            // batch norm bprop annotations are used to indicate if variance is in inverse stddev
            // format
            auto op_annotations =
                std::make_shared<ngraph::runtime::gpu::BatchNormBackpropAnnotations>();

            // pass must be run prior to GOE elimination
            // collect all batch norm inputs to batch norm backward op
            std::vector<std::shared_ptr<op::v0::BatchNormTraining>> bns;
            for (auto& arg : bnbp->get_arguments())
            {
                if (auto bn = std::dynamic_pointer_cast<op::v0::BatchNormTraining>(arg))
                {
                    bns.push_back(bn);
                }
            }

            // only replace if some of the inputs to backprop are from fprop directly
            if (bns.size())
            {
                if (auto target = std::dynamic_pointer_cast<op::v0::BatchNormTraining>(bns.front()))
                {
                    auto replacement = std::make_shared<op::gpu::BatchNormTrainingWithStats>(
                        target->get_eps_value(),
                        target->get_input_source_output(0),
                        target->get_input_source_output(1),
                        target->get_input_source_output(2));

                    // replace all users of old batchnorm with cudnn batchnorm
                    for (size_t i = 0; i < target->get_output_size(); i++)
                    {
                        Output<Node> target_output = target->output(i);
                        std::set<ngraph::Input<Node>> copy_inputs =
                            target_output.get_target_inputs();
                        for (auto input : copy_inputs)
                        {
                            input.replace_source_output(replacement->output(i));
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
