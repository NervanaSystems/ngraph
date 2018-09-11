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

// ==============================================[ onnxGetBackendInfo ] =======

namespace
{
    constexpr std::size_t g_default_info_value_size{50};

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
    }

    namespace device_type
    {
        std::string to_string(::onnxEnum device_type)
        {
#define ONNXIFI_DEVICE_TYPE_(value__)                                                              \
    case ONNXIFI_DEVICE_TYPE_##value__: return "ONNXIFI_DEVICE_TYPE_" #value__;

            switch (device_type)
            {
                ONNXIFI_DEVICE_TYPE_(NPU);
                ONNXIFI_DEVICE_TYPE_(DSP);
                ONNXIFI_DEVICE_TYPE_(GPU);
                ONNXIFI_DEVICE_TYPE_(CPU);
                ONNXIFI_DEVICE_TYPE_(FPGA);
                ONNXIFI_DEVICE_TYPE_(HETEROGENEOUS);
            default: return "UNKNOWN (" + std::to_string(device_type) + ")";
            }
        }
    }

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

    template <typename T>
    void backend_info_test_success(const std::vector<::onnxBackendID>& backend_ids,
                                   ::onnxBackendInfo infoType)
    {
        for (const auto& id : backend_ids)
        {
            std::size_t info_value_size{sizeof(T)};
            T info_value{0};
            ::onnxStatus status{::onnxGetBackendInfo(id, infoType, &info_value, &info_value_size)};
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }
        }
    }

    template <>
    void backend_info_test_success<char[]>(const std::vector<::onnxBackendID>& backend_ids,
                                           ::onnxBackendInfo infoType)
    {
        for (const auto& id : backend_ids)
        {
            std::size_t info_value_size{g_default_info_value_size};
            std::vector<char> info_value(g_default_info_value_size);
            ::onnxStatus status{
                ::onnxGetBackendInfo(id, infoType, info_value.data(), &info_value_size)};
            if (status == ONNXIFI_STATUS_FALLBACK)
            {
                info_value.resize(info_value_size);
                status = ::onnxGetBackendInfo(id, infoType, info_value.data(), &info_value_size);
            }
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }
        }
    }

    template <typename T>
    void backend_info_test_fallback(const std::vector<::onnxBackendID>& backend_ids,
                                    ::onnxBackendInfo infoType)
    {
        for (const auto& id : backend_ids)
        {
            std::size_t info_value_size{0};
            T info_value{0};
            ::onnxStatus status{::onnxGetBackendInfo(id, infoType, &info_value, &info_value_size)};
            if (status != ONNXIFI_STATUS_FALLBACK)
            {
                throw error::status{status, ONNXIFI_STATUS_FALLBACK};
            }
            if (info_value_size == 0)
            {
                throw std::runtime_error{"zero number of elements returned"};
            }
        }
    }

    template <>
    void backend_info_test_fallback<char[]>(const std::vector<::onnxBackendID>& backend_ids,
                                            ::onnxBackendInfo infoType)
    {
        for (const auto& backend_id : backend_ids)
        {
            std::size_t info_value_size{0};
            std::vector<char> info_value{};
            ::onnxStatus status{
                ::onnxGetBackendInfo(backend_id, infoType, info_value.data(), &info_value_size)};
            if (status != ONNXIFI_STATUS_FALLBACK)
            {
                throw error::status{status};
            }
            if (info_value_size == 0)
            {
                throw std::runtime_error{"zero number of elements returned"};
            }
        }
    }

    void backend_info_test_fallback_nullptr(const std::vector<::onnxBackendID>& backend_ids,
                                            ::onnxBackendInfo infoType)
    {
        for (const auto& backend_id : backend_ids)
        {
            std::size_t info_value_size{0};
            ::onnxStatus status{
                ::onnxGetBackendInfo(backend_id, infoType, nullptr, &info_value_size)};
            if (status != ONNXIFI_STATUS_FALLBACK)
            {
                throw error::status{status};
            }
            if (info_value_size == 0)
            {
                throw std::runtime_error{"zero number of elements returned"};
            }
        }
    }

    void backend_info_test_invalid_pointer(const std::vector<::onnxBackendID>& backend_ids,
                                           ::onnxBackendInfo infoType)
    {
        for (const auto& backend_id : backend_ids)
        {
            ::onnxStatus status{::onnxGetBackendInfo(backend_id, infoType, nullptr, nullptr)};
            if (status != ONNXIFI_STATUS_INVALID_POINTER)
            {
                throw error::status{status, ONNXIFI_STATUS_INVALID_POINTER};
            }
        }
    }

    template <typename T>
    void backend_info_test_result(const std::vector<::onnxBackendID>& backend_ids,
                                  ::onnxBackendInfo infoType,
                                  const std::function<bool(T, std::size_t)>& fn)
    {
        for (const auto& id : backend_ids)
        {
            std::size_t info_value_size{sizeof(T)};
            T info_value{};
            ::onnxStatus status{::onnxGetBackendInfo(id, infoType, &info_value, &info_value_size)};
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }
            if (!fn(info_value, info_value_size))
            {
                throw std::runtime_error{"received information does not match"};
            }
        }
    }

    template <>
    void backend_info_test_result<char[]>(const std::vector<::onnxBackendID>& backend_ids,
                                          ::onnxBackendInfo infoType,
                                          const std::function<bool(char[], std::size_t)>& fn)
    {
        for (const auto& id : backend_ids)
        {
            std::size_t info_value_size{0};
            ::onnxStatus status{::onnxGetBackendInfo(id, infoType, nullptr, &info_value_size)};
            if (status != ONNXIFI_STATUS_FALLBACK)
            {
                throw error::status{status, ONNXIFI_STATUS_FALLBACK};
            }
            std::vector<char> info_value(info_value_size);
            status = ::onnxGetBackendInfo(id, infoType, info_value.data(), &info_value_size);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }
            if (!fn(info_value.data(), info_value_size))
            {
                throw std::runtime_error{"received information does not match"};
            }
        }
    }

} // namespace {anonymous}

TEST(onnxifi, get_backend_info_invalid_id)
{
    std::size_t info_value_size{g_default_info_value_size};
    char info_value[g_default_info_value_size];
    ::onnxStatus status{
        ::onnxGetBackendInfo(nullptr, ONNXIFI_BACKEND_VERSION, info_value, &info_value_size)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_ID);
}

TEST(onnxifi, get_backend_info_unsupported_attribute)
{
    auto backend_ids = get_backend_ids();
    std::size_t info_value_size{g_default_info_value_size};
    char info_value[g_default_info_value_size];
    for (const auto& backend_id : backend_ids)
    {
        ::onnxStatus status{
            ::onnxGetBackendInfo(backend_id, 9999999, info_value, &info_value_size)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE);
    }
}

#define BACKEND_INFO_TEST_SUCCESS(type__, ids__, attribute__)                                      \
    backend_info_test_success<type__>(ids__, ONNXIFI_BACKEND_##attribute__);

#define BACKEND_INFO_TEST_FALLBACK(type__, ids__, attribute__)                                     \
    backend_info_test_fallback<type__>(ids__, ONNXIFI_BACKEND_##attribute__);

#define BACKEND_INFO_TEST_FALLBACK_NULL(ids__, attribute__)                                        \
    backend_info_test_fallback_nullptr(ids__, ONNXIFI_BACKEND_##attribute__);

#define BACKEND_INFO_TEST_INVALID_POINTER(ids__, attribute_)                                       \
    backend_info_test_invalid_pointer(ids__, ONNXIFI_BACKEND_##attribute_);

#define BACKEND_INFO_TEST_RESULT(type__, ids__, attribute__, function__)                           \
    backend_info_test_result<type__>(ids__, ONNXIFI_BACKEND_##attribute__, function__);

TEST(onnxifi, get_backend_info_onnxifi_version)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(uint64_t, ids, ONNXIFI_VERSION)
    BACKEND_INFO_TEST_FALLBACK(uint64_t, ids, ONNXIFI_VERSION)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, ONNXIFI_VERSION)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, ONNXIFI_VERSION)
    BACKEND_INFO_TEST_RESULT(
        uint64_t, ids, ONNXIFI_VERSION, [](uint64_t info_value, std::size_t info_value_size) {
            return (info_value == 1) && (info_value_size == sizeof(uint64_t));
        });
}

TEST(onnxifi, get_backend_info_name)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, NAME)
    BACKEND_INFO_TEST_FALLBACK(char[], ids, NAME)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, NAME)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, NAME)
    BACKEND_INFO_TEST_RESULT(
        char[], ids, NAME, [](const char* info_value, std::size_t info_value_size) {
            return /* TODO */ true;
        })
}

TEST(onnxifi, get_backend_info_vendor)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, VENDOR)
    BACKEND_INFO_TEST_FALLBACK(char[], ids, VENDOR)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, VENDOR)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, VENDOR)
    BACKEND_INFO_TEST_RESULT(
        char[], ids, VENDOR, [](const char* info_value, std::size_t info_value_size) {
            return /* TODO */ true;
        })
}

TEST(onnxifi, get_backend_info_version)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, VERSION)
    BACKEND_INFO_TEST_FALLBACK(char[], ids, VERSION)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, VERSION)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, VERSION)
    BACKEND_INFO_TEST_RESULT(
        char[], ids, VERSION, [](const char* info_value, std::size_t info_value_size) {
            return std::memcmp(info_value, NGRAPH_VERSION, info_value_size) == 0;
        });
}

TEST(onnxifi, get_backend_info_extensions)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, EXTENSIONS)
    BACKEND_INFO_TEST_FALLBACK(char[], ids, EXTENSIONS)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, EXTENSIONS)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, EXTENSIONS)
    BACKEND_INFO_TEST_RESULT(
        char[], ids, EXTENSIONS, [](const char* info_value, std::size_t info_value_size) {
            return /* TODO */ true;
        })
}

TEST(onnxifi, get_backend_info_device)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, DEVICE)
    BACKEND_INFO_TEST_FALLBACK(char[], ids, DEVICE)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, DEVICE)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, DEVICE)
    BACKEND_INFO_TEST_RESULT(
        char[], ids, DEVICE, [](const char* info_value, std::size_t info_value_size) {
            std::cout << "device: '" << std::string(info_value, info_value + info_value_size)
                      << "'\n";
            return true;
        })
}

TEST(onnxifi, get_backend_info_device_type)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(::onnxEnum, ids, DEVICE_TYPE)
    BACKEND_INFO_TEST_FALLBACK(::onnxEnum, ids, DEVICE_TYPE)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, DEVICE_TYPE)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, DEVICE_TYPE)
    BACKEND_INFO_TEST_RESULT(::onnxEnum,
                             ids,
                             DEVICE_TYPE,
                             [](::onnxEnum info_value, std::size_t info_value_size) -> bool {
                                 std::cout << "type: '" << device_type::to_string(info_value)
                                           << "'\n";
                                 return true;
                             })
}

TEST(onnxifi, get_backend_info_onnx_ir_version)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, ONNX_IR_VERSION)
    BACKEND_INFO_TEST_FALLBACK(char[], ids, ONNX_IR_VERSION)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, ONNX_IR_VERSION)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, ONNX_IR_VERSION)
    BACKEND_INFO_TEST_RESULT(
        char[], ids, ONNX_IR_VERSION, [](const char* info_value, std::size_t info_value_size) {
            return std::memcmp(info_value, ONNX_VERSION, info_value_size) == 0;
        })
}

TEST(onnxifi, get_backend_info_opset_version)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, OPSET_VERSION)
    BACKEND_INFO_TEST_FALLBACK(char[], ids, OPSET_VERSION)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, OPSET_VERSION)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, OPSET_VERSION)
    BACKEND_INFO_TEST_RESULT(
        char[], ids, OPSET_VERSION, [](const char* info_value, std::size_t info_value_size) {
            return std::memcmp(info_value, ONNX_OPSET_VERSION, info_value_size) == 0;
        })
}

TEST(onnxifi, get_backend_info_capabilities)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(::onnxBitfield, ids, CAPABILITIES)
    BACKEND_INFO_TEST_FALLBACK(::onnxBitfield, ids, CAPABILITIES)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, CAPABILITIES)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, CAPABILITIES)
    BACKEND_INFO_TEST_RESULT(::onnxBitfield,
                             ids,
                             CAPABILITIES,
                             [](::onnxBitfield info_value, std::size_t info_value_size) {
                                 return (info_value == ONNXIFI_CAPABILITY_THREAD_SAFE) &&
                                        (info_value_size == sizeof(::onnxBitfield));
                             })
}

TEST(onnxifi, get_backend_info_graph_init_properties)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(::onnxBitfield, ids, GRAPH_INIT_PROPERTIES)
    BACKEND_INFO_TEST_FALLBACK(::onnxBitfield, ids, GRAPH_INIT_PROPERTIES)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, GRAPH_INIT_PROPERTIES)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, GRAPH_INIT_PROPERTIES)
}

TEST(onnxifi, get_backend_info_synchronization_types)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(::onnxBitfield, ids, SYNCHRONIZATION_TYPES)
    BACKEND_INFO_TEST_FALLBACK(::onnxBitfield, ids, SYNCHRONIZATION_TYPES)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, SYNCHRONIZATION_TYPES)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, SYNCHRONIZATION_TYPES)
}

TEST(onnxifi, get_backend_info_memory_size)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(uint64_t, ids, MEMORY_SIZE)
    BACKEND_INFO_TEST_FALLBACK(uint64_t, ids, MEMORY_SIZE)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, MEMORY_SIZE)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, MEMORY_SIZE)
}

TEST(onnxifi, get_backend_info_max_graph_size)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(uint64_t, ids, MAX_GRAPH_SIZE)
    BACKEND_INFO_TEST_FALLBACK(uint64_t, ids, MAX_GRAPH_SIZE)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, MAX_GRAPH_SIZE)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, MAX_GRAPH_SIZE)
}

TEST(onnxifi, get_backend_info_max_graph_count)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(uint64_t, ids, MAX_GRAPH_COUNT)
    BACKEND_INFO_TEST_FALLBACK(uint64_t, ids, MAX_GRAPH_COUNT)
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, MAX_GRAPH_COUNT)
    BACKEND_INFO_TEST_INVALID_POINTER(ids, MAX_GRAPH_COUNT)
}

// ==================================================[ onnxInitBackend ]=======

TEST(onnxifi, init_backend)
{
    auto backend_ids = get_backend_ids();
    ::onnxBackend backend;
    for (const auto& backend_id : backend_ids)
    {
        ::onnxStatus status{::onnxInitBackend(backend_id, nullptr, &backend)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, init_backend_double_init)
{
    auto backend_ids = get_backend_ids();
    ::onnxStatus status;
    for (const auto& backend_id : backend_ids)
    {
        ::onnxBackend backend_a;
        status = ::onnxInitBackend(backend_id, nullptr, &backend_a);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        ::onnxBackend backend_b;
        status = ::onnxInitBackend(backend_id, nullptr, &backend_b);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        EXPECT_TRUE(backend_a == backend_b);
    }
}

// ONNXIFI_STATUS_INVALID_ID
// The function call failed because backendID is not an ONNXIFI backend ID.

TEST(onnxifi, init_backend_invalid_id)
{
    ::onnxBackend backend;
    ::onnxStatus status{::onnxInitBackend(nullptr, nullptr, &backend)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_ID);
    EXPECT_TRUE(backend == nullptr);
}

// ONNXIFI_STATUS_INVALID_POINTER
// The function call failed because backend pointer is NULL.

TEST(onnxifi, init_backend_invalid_pointer)
{
    auto backend_ids = get_backend_ids();
    for (const auto& id : backend_ids)
    {
        ::onnxStatus status{::onnxInitBackend(id, nullptr, nullptr)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);
    }
}
