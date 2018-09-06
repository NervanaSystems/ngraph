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

#include <cstring>

#include <gtest/gtest.h>
#include <onnxifi.h>

#include "ngraph/runtime/backend_manager.hpp"

const constexpr std::size_t g_backend_ids_count{10};

TEST(onnxifi, get_backend_ids)
{
    ::onnxBackendID backendIDs[g_backend_ids_count];
    std::size_t count{g_backend_ids_count};
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
    ::onnxBackendID backendIDs[g_backend_ids_count];
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
    ::onnxBackendID first_ids[g_backend_ids_count];
    std::size_t first_count{g_backend_ids_count};
    EXPECT_TRUE(::onnxGetBackendIDs(first_ids, &first_count) == ONNXIFI_STATUS_SUCCESS);
    EXPECT_TRUE(first_count == ngraph::runtime::BackendManager::get_registered_backends().size());
    ::onnxBackendID second_ids[g_backend_ids_count];
    std::size_t second_count{g_backend_ids_count};
    EXPECT_TRUE(::onnxGetBackendIDs(second_ids, &second_count) == ONNXIFI_STATUS_SUCCESS);
    EXPECT_TRUE(second_count == ngraph::runtime::BackendManager::get_registered_backends().size());
    EXPECT_TRUE(first_count == second_count);
    EXPECT_TRUE(std::memcmp(first_ids, second_ids, first_count) == 0);
}
