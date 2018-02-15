/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <ngraph/file_util.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/serializer.hpp>
namespace ngraph
{
    using CFrame = std::shared_ptr<runtime::CallFrame>;
    using TViews = std::vector<std::shared_ptr<runtime::TensorView>>;
    using CallFrameIO = std::tuple<CFrame, CFrame, TViews, TViews>;

    /// Create forward/backward call frame(s) and input/ouput TensorViews for given function.
    CallFrameIO
        get_cfio(std::string backend_type, std::shared_ptr<Function> f, bool backward = false)
    {
        auto manager = runtime::Manager::get(backend_type);
        auto external = manager->compile(f);
        auto backend = manager->allocate_backend();
        auto cf = backend->make_call_frame(external);
        auto result = backend->make_primary_tensor_view(f->get_output_element_type(0),
                                                        f->get_output_shape(0));
        std::vector<std::shared_ptr<runtime::TensorView>> viv;
        for (const auto& i : f->get_parameters())
            viv.push_back(backend->make_primary_tensor_view(i->get_element_type(), i->get_shape()));
        std::vector<std::shared_ptr<runtime::TensorView>> vrv;
        for (int i = 0; i < f->get_output_size(); ++i)
            vrv.push_back(backend->make_primary_tensor_view(f->get_output_element_type(i),
                                                            f->get_output_shape(i)));
        if (!backward)
            return CallFrameIO{cf, nullptr, viv, vrv};
        auto C =
            std::make_shared<op::Parameter>(f->get_output_element_type(0), f->get_output_shape(0));
        std::vector<std::shared_ptr<op::Parameter>> backparam;
        backparam.push_back(C);
        viv.push_back(backend->make_primary_tensor_view(C->get_element_type(), C->get_shape()));
        for (const auto& i : f->get_parameters())
            vrv.push_back(backend->make_primary_tensor_view(i->get_element_type(), i->get_shape()));
        std::vector<std::shared_ptr<Node>> dYdXs;
        auto op = f->get_result();
        for (const auto& i : f->get_parameters())
        {
            dYdXs.push_back(op->backprop_node(i, C));
            backparam.push_back(i);
        }
        auto bf = std::make_shared<Function>(dYdXs, backparam);
        auto backward_external = manager->compile(bf);
        auto bf_cf = backend->make_call_frame(backward_external);
        return CallFrameIO{cf, bf_cf, viv, vrv};
    }
}
