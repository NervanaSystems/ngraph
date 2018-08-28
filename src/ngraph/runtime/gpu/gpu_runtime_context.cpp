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

#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"

using namespace ngraph;
using namespace ngraph::runtime::gpu;

extern "C" void ngraph::runtime::gpu::start_stopwatch(GPURuntimeContext* ctx, size_t idx)
{
    ctx->stopwatch_pool->get(idx).start();
}

extern "C" void ngraph::runtime::gpu::stop_stopwatch(GPURuntimeContext* ctx, size_t idx)
{
    ctx->stopwatch_pool->get(idx).stop();
}
extern "C" size_t ngraph::runtime::gpu::count_stopwatch(GPURuntimeContext* ctx, size_t idx)
{
    return ctx->stopwatch_pool->get(idx).get_call_count();
}
extern "C" size_t ngraph::runtime::gpu::us_stopwatch(GPURuntimeContext* ctx, size_t idx)
{
    return ctx->stopwatch_pool->get(idx).get_total_microseconds();
}
