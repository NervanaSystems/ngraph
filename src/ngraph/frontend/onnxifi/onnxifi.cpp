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

#include <cstdint>
#include <onnxifi.h>

extern "C" {

onnxStatus onnxGetBackentIDs(onnxBackendID* backendIDs, std::size_t* numBackends)
{
    return ONNXIFI_STATUS_INTERNAL_ERROR;
}

onnxStatus onnxReleaseBackendID(onnxBackendID backendID)
{
    return ONNXIFI_STATUS_INTERNAL_ERROR;
}

onnxStatus onnxGetBackendInfo(onnxBackendID backendID,
                              onnxBackendInfo infoType,
                              void* infoValue,
                              std::size_t* infoValueSize)
{
    return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
}

onnxStatus onnxGetBackendCompatibility(onnxBackendID backendID,
                                       std::size_t onnxModelSize,
                                       const void* onnxModel)
{
    return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
}

onnxStatus onnxInitBackend(onnxBackendID backendID,
                           const uint64_t* auxPropertiesList,
                           onnxBackend* backend)
{
    return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
}

onnxStatus onnxReleaseBackend(onnxBackend backend)
{
    return ONNXIFI_STATUS_INTERNAL_ERROR;
}

onnxStatus onnxInitEvent(onnxBackend backend, onnxEvent* event)
{
    return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
}

onnxStatus onnxSignalEvent(onnxEvent event)
{
    return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
}

onnxStatus onnxWaitEvent(onnxEvent event)
{
    return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
}

onnxStatus onnxReleaseEvent(onnxEvent event)
{
    return ONNXIFI_STATUS_INTERNAL_ERROR;
}

onnxStatus onnxInitGraph(onnxBackend backend,
                         std::size_t onnxModelSize,
                         const void* onnxModel,
                         std::uint32_t weightsCount,
                         const onnxTensorDescriptor* weightDescriptors,
                         onnxGraph* graph)
{
    return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
}

onnxStatus onnxSetGraphIO(onnxGraph graph,
                          std::uint32_t inputsCount,
                          const onnxTensorDescriptor* inputDescriptors,
                          std::uint32_t outputsCount,
                          const onnxTensorDescriptor* outputDescriptors)
{
    return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
}

onnxStatus
    onnxRunGraph(onnxGraph graph, const onnxMemoryFence* inputFence, onnxMemoryFence* outputFence)
{
    return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
}

onnxStatus onnxReleaseGraph(onnxGraph graph)
{
    return ONNXIFI_STATUS_INTERNAL_ERROR;
}

} /* extern "C" */
