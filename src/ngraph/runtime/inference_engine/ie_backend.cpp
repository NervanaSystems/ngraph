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

#if defined(NGRAPH_TBB_ENABLE)
#include <tbb/tbb_stddef.h>
#endif

#include "ie_backend.hpp"

#include "ngraph/component_manager.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/inference_engine/ie_backend_visibility.hpp"
#include "ngraph/runtime/inference_engine/ie_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

ngraph::runtime::inference_engine::IE_Backend::IE_Backend(const string& configuration_string)
{
    string config = configuration_string;
    // Get device name, after colon if present: IE:CPU -> CPU
    auto separator = config.find(":");
    if (separator != config.npos)
    {
        config = config.substr(separator);
    }
    device = config;
}

shared_ptr<ngraph::runtime::Tensor> ngraph::runtime::inference_engine::IE_Backend::create_tensor(
    const ngraph::element::Type& element_type, const ngraph::Shape& shape)
{
    return make_shared<IETensorView>(element_type, shape);
}

shared_ptr<ngraph::runtime::Executable>
    ngraph::runtime::inference_engine::IE_Backend::compile(shared_ptr<Function> func,
                                                           bool /* enable_performance_data */)
{
    return make_shared<IE_Executable>(func, device);
}

bool ngraph::runtime::inference_engine::IE_Backend::is_supported(const Node& node) const
{
    return true;
}

bool ngraph::runtime::inference_engine::IE_Backend::is_supported_property(
    const Property /* prop */) const
{
    return false;
}

extern "C" IE_BACKEND_API void ngraph_register_ie_backend()
{
    ngraph::runtime::BackendManager::register_backend("INFERENCE_ENGINE", [](const string& config) {
        return make_shared<ngraph::runtime::inference_engine::IE_Backend>(config);
    });
}
