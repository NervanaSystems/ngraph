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

#include <iterator>

#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/dyn_replace_slice.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/range.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/transpose.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/convert_opset_1_to_0.hpp"
#include "ngraph/pass/convert_opset_3_to_1.hpp"
#include "ngraph/pass/dyn_elimination.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/shape_relevance.hpp"
#include "ngraph/runtime/dynamic/dynamic_executable.hpp"
#include "ngraph/runtime/dynamic/dynamic_tensor.hpp"
#include "ngraph/specialize_function.hpp"

using namespace std;
using namespace ngraph;

runtime::dynamic::DynamicExecutable::DynamicExecutable(shared_ptr<Function> wrapped_function,
                                                       shared_ptr<runtime::Backend> wrapped_backend,
                                                       bool enable_performance_collection)
    : m_wrapped_function(wrapped_function)
    , m_wrapped_backend(wrapped_backend)
    , m_enable_performance_collection(enable_performance_collection)
{
    pass::Manager passes;
    passes.register_pass<pass::ShapeRelevance>();
    passes.run_passes(m_wrapped_function);

    set_parameters_and_results(*wrapped_function);
}

// Due to clang++-3.9 bugs, this needs to be a non-static separate function from
// count_dyn_nodes.
bool is_dynamic_op(const std::shared_ptr<Node>& op)
{
    return is_type<op::v1::Transpose>(op) || is_type<op::v0::DynBroadcast>(op) ||
           is_type<op::v0::DynReplaceSlice>(op) || is_type<op::v0::DynSlice>(op) ||
           is_type<op::v1::Reshape>(op) || is_type<op::v0::Range>(op) ||
           is_type<op::v1::ConvolutionBackpropData>(op) ||
           is_type<op::v1::ConvolutionBackpropFilters>(op) ||
           is_type<op::v1::AvgPoolBackprop>(op) || is_type<op::v1::Broadcast>(op) ||
           is_type<op::v3::Broadcast>(op) || is_type<op::v1::GenerateMask>(op);
}

// Helper for a vile hack in DynamicExecutable::call. See body of that function for details.
static size_t count_dyn_nodes(const shared_ptr<ngraph::Function>& f)
{
    size_t count = 0;
    for (auto op : f->get_ops())
    {
        if (is_dynamic_op(op))
        {
            count++;
        }
    }
    return count;
}

bool runtime::dynamic::DynamicExecutable::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    // TODO: Get cached executable out if it exists.
    // We will cache on:
    // (1) all shapes;
    // (2) all values of shape-relevant input tensors.

    std::vector<int> merged_input_shapes;
    std::ostringstream key;
    size_t loop_count = 0;
    for (auto& input : inputs)
    {
        if (m_wrapped_function->get_parameters()[loop_count]->is_relevant_to_shapes())
        {
            // Caching on values of Shape relevant inputs
            int size = input->get_size_in_bytes() / (input->get_element_type().bitwidth() / 8);
            std::vector<int64_t> data(size);
            input->read(data.data(), input->get_size_in_bytes());
            for (int i = 0; i < input->get_element_count(); i++)
            {
                merged_input_shapes.emplace_back(data[i]);
            }
        }
        else
        {
            // Caching on all remaining shapes
            for (int i = 0; i < input->get_shape().size(); i++)
            {
                merged_input_shapes.emplace_back(input->get_shape()[i]);
            }
        }
        // -1 is the separator.
        // So if shape of Input 1 = {2, 2, 3, 3} & Input 2 = {4, 5}
        // the key would be 2, 2, 3, 3, -1, 4, 5, -1
        merged_input_shapes.emplace_back(-1);
        loop_count++;
    }

    std::copy(merged_input_shapes.begin(),
              merged_input_shapes.end(),
              std::ostream_iterator<int>(key, ", "));

    if (m_cache->is_cached(merged_input_shapes))
    {
        std::vector<std::shared_ptr<runtime::Tensor>> wrapped_inputs;
        std::vector<std::shared_ptr<runtime::Tensor>> wrapped_outputs;

        std::shared_ptr<Function> clone = m_cache->get_cloned_function(merged_input_shapes);
        const ResultVector& results = clone->get_results();
        for (auto& result : results)
        {
            NGRAPH_CHECK(result->get_output_partial_shape(0).is_static(),
                         "Shape staticization failed for result node ",
                         *result);
        }
        NGRAPH_CHECK(results.size() == outputs.size());

        for (size_t i = 0; i < outputs.size(); i++)
        {
            if (auto dynamic_tensor =
                    std::dynamic_pointer_cast<runtime::dynamic::DynamicTensor>(outputs[i]))
            {
                dynamic_tensor->make_storage(results[i]->get_output_element_type(0),
                                             results[i]->get_output_shape(0));
                wrapped_outputs.push_back(dynamic_tensor->get_wrapped_tensor());
            }
            else
            {
                wrapped_outputs.push_back(outputs[i]);
            }
        }

        return m_cache->get_cached_entry(merged_input_shapes)->call(wrapped_outputs, inputs);
    }
    else
    {
        NGRAPH_CHECK(m_wrapped_function->get_parameters().size() == inputs.size());

        std::vector<std::shared_ptr<runtime::Tensor>> wrapped_inputs;
        std::vector<element::Type> arg_element_types;
        std::vector<PartialShape> arg_shapes;

        std::shared_ptr<Function> clone;
        {
            // We'll use AlignedBuffers to back the base pointers, storing them in this vector for
            // RAII
            // purposes.
            std::vector<AlignedBuffer> arg_buffers;
            arg_buffers.reserve(inputs.size());
            std::vector<void*> arg_value_base_pointers(inputs.size());

            size_t i = 0;

            for (auto& input : inputs)
            {
                if (m_wrapped_function->get_parameters()[i]->is_relevant_to_shapes())
                {
                    // TODO(amprocte): Move has_storage() to runtime::Tensor?
                    if (auto dynamic_tensor =
                            std::dynamic_pointer_cast<runtime::dynamic::DynamicTensor>(input))
                    {
                        NGRAPH_CHECK(dynamic_tensor->has_storage());
                    }

                    arg_buffers.emplace_back(input->get_size_in_bytes(), /*alignment=*/64);
                    arg_value_base_pointers[i] = arg_buffers.back().get_ptr();

                    // TODO(amprocte): For host-resident tensors we should be able to skip the read,
                    // but no API for that yet.
                    input->read(arg_value_base_pointers[i], input->get_size_in_bytes());
                }
                else
                {
                    arg_value_base_pointers[i] = nullptr;
                }

                if (auto dynamic_tensor =
                        std::dynamic_pointer_cast<runtime::dynamic::DynamicTensor>(input))
                {
                    NGRAPH_CHECK(dynamic_tensor->has_storage());
                    arg_element_types.push_back(
                        dynamic_tensor->get_wrapped_tensor()->get_element_type());
                    arg_shapes.push_back(dynamic_tensor->get_wrapped_tensor()->get_shape());
                    wrapped_inputs.push_back(dynamic_tensor->get_wrapped_tensor());
                }
                else
                {
                    arg_element_types.push_back(input->get_element_type());
                    arg_shapes.push_back(input->get_shape());
                    wrapped_inputs.push_back(input);
                }

                i++;
            }

            clone = specialize_function(
                m_wrapped_function, arg_element_types, arg_shapes, arg_value_base_pointers);
        }

        pass::Manager passes;
        // ConvertOpset3To1 should be moved below DynElimination
        // when ConstantFolding for v3 ops will be ready
        passes.register_pass<pass::ConvertOpset3To1>();
        passes.register_pass<pass::ConstantFolding>();
        passes.register_pass<pass::DynElimination>();
        passes.register_pass<pass::ConvertOpset1To0>(); // Converts dynamic v1 variants to v0 ops
        passes.set_per_pass_validation(false);

        // FIXME(amprocte): Vile, temporary hack: we need to do repeated rounds of
        // ConstantFolding/DynElimination until everything that DynElimination is supposed to
        // eliminate has actually been eliminated. We could do this by monitoring the return values
        // of the passes (keep iterating until both CF and DE report no changes), but that did not
        // seem to work so here we are. Probably a better fix is to somehow combine the matchers in
        // CF
        // and DE into one pass.
        size_t num_dyn_nodes_last_pass = std::numeric_limits<size_t>::max();

        while (num_dyn_nodes_last_pass != 0)
        {
            passes.run_passes(clone);
            auto num_dyn_nodes_this_pass = count_dyn_nodes(clone);

            NGRAPH_CHECK(num_dyn_nodes_this_pass < num_dyn_nodes_last_pass,
                         "Could not eliminate all Dyn nodes (",
                         num_dyn_nodes_this_pass,
                         " remaining)");

            num_dyn_nodes_last_pass = num_dyn_nodes_this_pass;
        }

        pass::Manager pass_val;
        pass_val.register_pass<pass::Validate>();
        pass_val.run_passes(clone);

        std::vector<std::shared_ptr<runtime::Tensor>> wrapped_outputs;

        const ResultVector& results = clone->get_results();
        for (auto& result : results)
        {
            NGRAPH_CHECK(result->get_output_partial_shape(0).is_static(),
                         "Shape staticization failed for result node ",
                         *result);
        }
        NGRAPH_CHECK(results.size() == outputs.size());

        for (size_t i = 0; i < outputs.size(); i++)
        {
            if (auto dynamic_tensor =
                    std::dynamic_pointer_cast<runtime::dynamic::DynamicTensor>(outputs[i]))
            {
                dynamic_tensor->make_storage(results[i]->get_output_element_type(0),
                                             results[i]->get_output_shape(0));
                wrapped_outputs.push_back(dynamic_tensor->get_wrapped_tensor());
            }
            else
            {
                wrapped_outputs.push_back(outputs[i]);
            }
        }

        auto compiled_executable =
            m_wrapped_backend->compile(clone, m_enable_performance_collection);
        // Put compiled executable in the cache.
        m_cache->add_entry(merged_input_shapes, compiled_executable, clone);
        auto result = compiled_executable->call(wrapped_outputs, wrapped_inputs);

        return result;
    }
}
