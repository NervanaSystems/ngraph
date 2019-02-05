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

#include <cstring>

#include <gtest/gtest.h>
#include <onnxifi.h>

#include "ngraph/runtime/backend_manager.hpp"

// ===============================================[ onnxGetBackendIDs ] =======

constexpr std::size_t g_default_backend_ids_count{10};

TEST(onnxifi, get_backend_ids)
{
    ::onnxBackendID backendIDs[g_default_backend_ids_count];
    std::size_t count{g_default_backend_ids_count};
    ::onnxStatus status{::onnxGetBackendIDs(backendIDs, &count)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    EXPECT_TRUE(count == ngraph::runtime::BackendManager::get_registered_backends().size());
}

TEST(onnxifi, get_backend_ids_buffer_null)
{
    std::size_t count{0};
    ::onnxStatus status{::onnxGetBackendIDs(nullptr, &count)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_FALLBACK);
    EXPECT_TRUE(count == ngraph::runtime::BackendManager::get_registered_backends().size());
}

TEST(onnxifi, get_backend_ids_count_null)
{
    ::onnxBackendID backendIDs[g_default_backend_ids_count];
    ::onnxStatus status{::onnxGetBackendIDs(backendIDs, nullptr)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);
}

TEST(onnxifi, get_backend_ids_null)
{
    ::onnxStatus status{::onnxGetBackendIDs(nullptr, nullptr)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);
}

TEST(onnxifi, get_backend_ids_consistency_check)
{
    ::onnxBackendID first_ids[g_default_backend_ids_count];
    std::size_t first_count{g_default_backend_ids_count};
    EXPECT_TRUE(::onnxGetBackendIDs(first_ids, &first_count) == ONNXIFI_STATUS_SUCCESS);
    EXPECT_TRUE(first_count == ngraph::runtime::BackendManager::get_registered_backends().size());
    ::onnxBackendID second_ids[g_default_backend_ids_count];
    std::size_t second_count{g_default_backend_ids_count};
    EXPECT_TRUE(::onnxGetBackendIDs(second_ids, &second_count) == ONNXIFI_STATUS_SUCCESS);
    EXPECT_TRUE(second_count == ngraph::runtime::BackendManager::get_registered_backends().size());
    EXPECT_TRUE(first_count == second_count);
    EXPECT_TRUE(std::memcmp(first_ids, second_ids, first_count) == 0);
}

// ============================================================================

namespace
{
    namespace status
    {
        std::string to_string(::onnxStatus status)
        {
#define ONNXIFI_STATUS_(value__)                                                                   \
    case ONNXIFI_STATUS_##value__: return "ONNXIFI_STATUS_" #value__;

            switch (status)
            {
                ONNXIFI_STATUS_(SUCCESS);
                ONNXIFI_STATUS_(FALLBACK);
                ONNXIFI_STATUS_(INVALID_ID);
                ONNXIFI_STATUS_(INVALID_SIZE);
                ONNXIFI_STATUS_(INVALID_POINTER);
                ONNXIFI_STATUS_(INVALID_PROTOBUF);
                ONNXIFI_STATUS_(INVALID_MODEL);
                ONNXIFI_STATUS_(INVALID_BACKEND);
                ONNXIFI_STATUS_(INVALID_GRAPH);
                ONNXIFI_STATUS_(INVALID_EVENT);
                ONNXIFI_STATUS_(INVALID_STATE);
                ONNXIFI_STATUS_(INVALID_NAME);
                ONNXIFI_STATUS_(INVALID_SHAPE);
                ONNXIFI_STATUS_(INVALID_DATATYPE);
                ONNXIFI_STATUS_(INVALID_MEMORY_TYPE);
                ONNXIFI_STATUS_(INVALID_MEMORY_LOCATION);
                ONNXIFI_STATUS_(INVALID_FENCE_TYPE);
                ONNXIFI_STATUS_(INVALID_PROPERTY);
                ONNXIFI_STATUS_(UNSUPPORTED_TAG);
                ONNXIFI_STATUS_(UNSUPPORTED_VERSION);
                ONNXIFI_STATUS_(UNSUPPORTED_OPERATOR);
                ONNXIFI_STATUS_(UNSUPPORTED_ATTRIBUTE);
                ONNXIFI_STATUS_(UNSUPPORTED_SHAPE);
                ONNXIFI_STATUS_(UNSUPPORTED_DATATYPE);
                ONNXIFI_STATUS_(NO_SYSTEM_MEMORY);
                ONNXIFI_STATUS_(NO_DEVICE_MEMORY);
                ONNXIFI_STATUS_(NO_SYSTEM_RESOURCES);
                ONNXIFI_STATUS_(NO_DEVICE_RESOURCES);
                ONNXIFI_STATUS_(BACKEND_UNAVAILABLE);
                ONNXIFI_STATUS_(INTERNAL_ERROR);
            default: return "UNKNOWN (" + std::to_string(status) + ")";
            }
        }

    } // namespace status

    namespace error
    {
        struct status : std::runtime_error
        {
            explicit status(::onnxStatus status, ::onnxStatus expected = ONNXIFI_STATUS_SUCCESS)
                : std::runtime_error{::status::to_string(status) +
                                     ": unexpected status; expected " +
                                     ::status::to_string(expected)}
            {
            }
        };

    } // namespace error

    std::vector<::onnxBackendID> get_backend_ids()
    {
        std::size_t count{g_default_backend_ids_count};
        ::onnxStatus status{::onnxGetBackendIDs(nullptr, &count)};
        if (status != ONNXIFI_STATUS_FALLBACK)
        {
            throw error::status{status, ONNXIFI_STATUS_FALLBACK};
        }
        std::vector<::onnxBackendID> backend_ids(count);
        status = ::onnxGetBackendIDs(backend_ids.data(), &count);
        if (status == ONNXIFI_STATUS_FALLBACK)
        {
            backend_ids.resize(count);
            status = ::onnxGetBackendIDs(backend_ids.data(), &count);
        }
        if (status != ONNXIFI_STATUS_SUCCESS)
        {
            throw error::status{status};
        }
        if (backend_ids.empty())
        {
            throw std::runtime_error{"no backends registered"};
        }
        return backend_ids;
    }

} // namespace <anonymous>

// =============================================[ onnxReleaseBackendID ]=======

TEST(onnxifi, release_backend_id)
{
    auto backend_ids = get_backend_ids();
    for (auto& backend_id : backend_ids)
    {
        ::onnxStatus status{::onnxReleaseBackendID(backend_id)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, release_backend_id_invalid_id)
{
    ::onnxStatus status{::onnxReleaseBackendID(nullptr)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_ID);
}
