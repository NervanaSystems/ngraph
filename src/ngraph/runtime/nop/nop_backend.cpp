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

#include "ngraph/runtime/nop/nop_backend_visibility.hpp"

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/nop/nop_backend.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

extern "C" NOP_BACKEND_API void ngraph_register_nop_backend()
{
    runtime::BackendManager::register_backend("NOP", [](const std::string& /* config */) {
        return std::make_shared<runtime::nop::NOPBackend>();
    });
}

shared_ptr<runtime::Tensor> runtime::nop::NOPBackend::create_tensor(const element::Type& type,
                                                                    const Shape& shape)
{
    return make_shared<runtime::HostTensor>(type, shape, "external");
}

shared_ptr<runtime::Tensor> runtime::nop::NOPBackend::create_tensor(const element::Type& type,
                                                                    const Shape& shape,
                                                                    void* memory_pointer)
{
    return make_shared<runtime::HostTensor>(type, shape, memory_pointer, "external");
}

shared_ptr<runtime::Executable>
    runtime::nop::NOPBackend::compile(shared_ptr<Function> function,
                                      bool enable_performance_collection)
{
    return make_shared<NOPExecutable>(function, enable_performance_collection);
}

runtime::nop::NOPExecutable::NOPExecutable(shared_ptr<Function> function,
                                           bool /* enable_performance_collection */)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignLayout<DenseTensorLayout>>();
    pass_manager.run_passes(function);

    set_parameters_and_results(*function);
}

bool runtime::nop::NOPExecutable::call(const vector<shared_ptr<runtime::Tensor>>& /* outputs */,
                                       const vector<shared_ptr<runtime::Tensor>>& /* inputs */)
{
    return true;
}
