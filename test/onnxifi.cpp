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
#include <fstream>
#include <map>
#include <utility>
#include <vector>
#include <thread>

#include <gtest/gtest.h>
#include <onnxifi.h>

#include "ngraph/runtime/backend_manager.hpp"
#include "util/ndarray.hpp"
#include "util/all_close_f.hpp"

// ================================================[ onnxGetBackendIDs ]=======

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

// =============================================[ onnxReleaseBackendID ]=======

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

// ===============================================[ onnxGetBackendInfo ]=======

namespace
{
    constexpr std::size_t g_default_info_value_size{50};

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
    backend_info_test_success<type__>(ids__, ONNXIFI_BACKEND_##attribute__)

#define BACKEND_INFO_TEST_FALLBACK(type__, ids__, attribute__)                                     \
    backend_info_test_fallback<type__>(ids__, ONNXIFI_BACKEND_##attribute__)

#define BACKEND_INFO_TEST_FALLBACK_NULL(ids__, attribute__)                                        \
    backend_info_test_fallback_nullptr(ids__, ONNXIFI_BACKEND_##attribute__)

#define BACKEND_INFO_TEST_INVALID_POINTER(ids__, attribute_)                                       \
    backend_info_test_invalid_pointer(ids__, ONNXIFI_BACKEND_##attribute_)

#define BACKEND_INFO_TEST_RESULT(type__, ids__, attribute__, function__)                           \
    backend_info_test_result<type__>(ids__, ONNXIFI_BACKEND_##attribute__, function__)

TEST(onnxifi, get_backend_info_onnxifi_version)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(uint64_t, ids, ONNXIFI_VERSION);
    BACKEND_INFO_TEST_FALLBACK(uint64_t, ids, ONNXIFI_VERSION);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, ONNXIFI_VERSION);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, ONNXIFI_VERSION);
    BACKEND_INFO_TEST_RESULT(
        uint64_t, ids, ONNXIFI_VERSION, [](uint64_t info_value, std::size_t info_value_size) {
            return (info_value == 1) && (info_value_size == sizeof(uint64_t));
        });
}

TEST(onnxifi, get_backend_info_name)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, NAME);
    BACKEND_INFO_TEST_FALLBACK(char[], ids, NAME);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, NAME);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, NAME);
    BACKEND_INFO_TEST_RESULT(
        char[], ids, NAME, [](const char* info_value, std::size_t info_value_size) {
            return /* TODO */ true;
        });
}

TEST(onnxifi, get_backend_info_vendor)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, VENDOR);
    BACKEND_INFO_TEST_FALLBACK(char[], ids, VENDOR);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, VENDOR);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, VENDOR);
    BACKEND_INFO_TEST_RESULT(
        char[], ids, VENDOR, [](const char* info_value, std::size_t info_value_size) {
            return /* TODO */ true;
        });
}

TEST(onnxifi, get_backend_info_version)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, VERSION);
    BACKEND_INFO_TEST_FALLBACK(char[], ids, VERSION);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, VERSION);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, VERSION);
    BACKEND_INFO_TEST_RESULT(
        char[], ids, VERSION, [](const char* info_value, std::size_t info_value_size) {
            return std::memcmp(info_value, NGRAPH_VERSION, info_value_size) == 0;
        });
}

TEST(onnxifi, get_backend_info_extensions)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, EXTENSIONS);
    BACKEND_INFO_TEST_FALLBACK(char[], ids, EXTENSIONS);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, EXTENSIONS);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, EXTENSIONS);
    BACKEND_INFO_TEST_RESULT(
        char[], ids, EXTENSIONS, [](const char* info_value, std::size_t info_value_size) {
            return /* TODO */ true;
        });
}

TEST(onnxifi, get_backend_info_device)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, DEVICE);
    BACKEND_INFO_TEST_FALLBACK(char[], ids, DEVICE);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, DEVICE);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, DEVICE);
    BACKEND_INFO_TEST_RESULT(
        char[], ids, DEVICE, [](const char* info_value, std::size_t info_value_size) {
            std::cout << "device: '" << std::string(info_value, info_value + info_value_size)
                      << "'\n";
            return true;
        });
}

TEST(onnxifi, get_backend_info_device_type)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(::onnxEnum, ids, DEVICE_TYPE);
    BACKEND_INFO_TEST_FALLBACK(::onnxEnum, ids, DEVICE_TYPE);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, DEVICE_TYPE);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, DEVICE_TYPE);
    BACKEND_INFO_TEST_RESULT(::onnxEnum,
                             ids,
                             DEVICE_TYPE,
                             [](::onnxEnum info_value, std::size_t info_value_size) -> bool {
                                 std::cout << "type: '" << device_type::to_string(info_value)
                                           << "'\n";
                                 return true;
                             });
}

TEST(onnxifi, get_backend_info_onnx_ir_version)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, ONNX_IR_VERSION);
    BACKEND_INFO_TEST_FALLBACK(char[], ids, ONNX_IR_VERSION);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, ONNX_IR_VERSION);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, ONNX_IR_VERSION);
    BACKEND_INFO_TEST_RESULT(
        char[], ids, ONNX_IR_VERSION, [](const char* info_value, std::size_t info_value_size) {
            return std::memcmp(info_value, ONNX_VERSION, info_value_size) == 0;
        });
}

TEST(onnxifi, get_backend_info_opset_version)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(char[], ids, OPSET_VERSION);
    BACKEND_INFO_TEST_FALLBACK(char[], ids, OPSET_VERSION);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, OPSET_VERSION);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, OPSET_VERSION);
    BACKEND_INFO_TEST_RESULT(
        char[], ids, OPSET_VERSION, [](const char* info_value, std::size_t info_value_size) {
            return std::memcmp(info_value, ONNX_OPSET_VERSION, info_value_size) == 0;
        });
}

TEST(onnxifi, get_backend_info_capabilities)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(::onnxBitfield, ids, CAPABILITIES);
    BACKEND_INFO_TEST_FALLBACK(::onnxBitfield, ids, CAPABILITIES);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, CAPABILITIES);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, CAPABILITIES);
    BACKEND_INFO_TEST_RESULT(::onnxBitfield,
                             ids,
                             CAPABILITIES,
                             [](::onnxBitfield info_value, std::size_t info_value_size) {
                                 return (info_value == ONNXIFI_CAPABILITY_THREAD_SAFE) &&
                                        (info_value_size == sizeof(::onnxBitfield));
                             });
}

TEST(onnxifi, get_backend_info_graph_init_properties)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(::onnxBitfield, ids, GRAPH_INIT_PROPERTIES);
    BACKEND_INFO_TEST_FALLBACK(::onnxBitfield, ids, GRAPH_INIT_PROPERTIES);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, GRAPH_INIT_PROPERTIES);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, GRAPH_INIT_PROPERTIES);
}

TEST(onnxifi, get_backend_info_synchronization_types)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(::onnxBitfield, ids, SYNCHRONIZATION_TYPES);
    BACKEND_INFO_TEST_FALLBACK(::onnxBitfield, ids, SYNCHRONIZATION_TYPES);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, SYNCHRONIZATION_TYPES);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, SYNCHRONIZATION_TYPES);
}

TEST(onnxifi, get_backend_info_memory_size)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(uint64_t, ids, MEMORY_SIZE);
    BACKEND_INFO_TEST_FALLBACK(uint64_t, ids, MEMORY_SIZE);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, MEMORY_SIZE);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, MEMORY_SIZE);
}

TEST(onnxifi, get_backend_info_max_graph_size)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(uint64_t, ids, MAX_GRAPH_SIZE);
    BACKEND_INFO_TEST_FALLBACK(uint64_t, ids, MAX_GRAPH_SIZE);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, MAX_GRAPH_SIZE);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, MAX_GRAPH_SIZE);
}

TEST(onnxifi, get_backend_info_max_graph_count)
{
    auto ids = get_backend_ids();
    BACKEND_INFO_TEST_SUCCESS(uint64_t, ids, MAX_GRAPH_COUNT);
    BACKEND_INFO_TEST_FALLBACK(uint64_t, ids, MAX_GRAPH_COUNT);
    BACKEND_INFO_TEST_FALLBACK_NULL(ids, MAX_GRAPH_COUNT);
    BACKEND_INFO_TEST_INVALID_POINTER(ids, MAX_GRAPH_COUNT);
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

TEST(onnxifi, init_backend_invalid_id)
{
    ::onnxBackend backend;
    ::onnxStatus status{::onnxInitBackend(nullptr, nullptr, &backend)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_ID);
    EXPECT_TRUE(backend == nullptr);
}

TEST(onnxifi, init_backend_invalid_pointer)
{
    auto backend_ids = get_backend_ids();
    for (const auto& id : backend_ids)
    {
        ::onnxStatus status{::onnxInitBackend(id, nullptr, nullptr)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);
    }
}

// ===============================================[ onnxReleaseBackend ]=======

namespace
{
    class InitializedBackends : public std::vector<::onnxBackend>
    {
    public:
        InitializedBackends()
        {
            auto backend_ids = get_backend_ids();
            for (const auto& backend_id : backend_ids)
            {
                ::onnxBackend backend;
                ::onnxStatus status{::onnxInitBackend(backend_id, nullptr, &backend)};
                if (status != ONNXIFI_STATUS_SUCCESS)
                {
                    throw error::status{status};
                }
                push_back(backend);
            }
        }
        ~InitializedBackends()
        {
            for (const auto& backend : *this)
            {
                // Read the status, but ignore it. For some tests the backends
                // might be already released.
                ::onnxStatus status{::onnxReleaseBackend(backend)};
            }
        }
    };

    namespace
    {
        std::vector<char> load_model(const std::string& name)
        {
            std::ifstream sin{SERIALIZED_ZOO "/onnx/" + name, std::ios::ate | std::ios::binary};
            if (!sin.is_open())
            {
                throw std::runtime_error{"unable to load the model"};
            }
            std::ifstream::pos_type size{sin.tellg()};
            std::vector<char> model(size);
            sin.seekg(0, std::ios::beg);
            sin.read(model.data(), size);
            return model;
        }

        std::vector<char> load_model()
        {
            return load_model("add_abc.onnx");
        }

    } // namespace <anonymous>

}

TEST(onnxifi, release_backend)
{
    InitializedBackends backends{};
    for (auto& backend : backends)
    {
        ::onnxStatus status{::onnxReleaseBackend(backend)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, release_backend_invalid_backend)
{
    ::onnxStatus status{::onnxReleaseBackend(nullptr)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_BACKEND);
}

// ====================================================[ onnxInitEvent ]=======

TEST(onnxifi, init_event_invalid_backend)
{
    ::onnxEvent event;
    ::onnxStatus status{::onnxInitEvent(nullptr, &event)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_BACKEND);
    EXPECT_TRUE(event == nullptr);
}

TEST(onnxifi, init_event_invalid_pointer)
{
    InitializedBackends backends{};
    for (const auto& backend : backends)
    {
        ::onnxStatus status{::onnxInitEvent(backend, nullptr)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);
    }
}

// =================================================[ onnxReleaseEvent ]=======

TEST(onnxifi, release_event)
{
    InitializedBackends backends{};
    for (const auto& backend : backends)
    {
        ::onnxEvent event;
        ::onnxStatus status{::onnxInitEvent(backend, &event)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        status = ::onnxReleaseEvent(event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, release_event_invalid_event)
{
    ::onnxStatus status{::onnxReleaseEvent(nullptr)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
}

// ==================================================[ onnxSignalEvent ]=======

TEST(onnxifi, signal_event)
{
    InitializedBackends backends{};
    for (const auto& backend : backends)
    {
        ::onnxEvent event;
        ::onnxStatus status{::onnxInitEvent(backend, &event)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        status = ::onnxSignalEvent(event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        status = ::onnxReleaseEvent(event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, signal_event_invalid_event)
{
    ::onnxStatus status{::onnxSignalEvent(nullptr)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_EVENT);
}

TEST(onnxifi, signal_event_invalid_state)
{
    InitializedBackends backends{};
    for (const auto& backend : backends)
    {
        ::onnxEvent event;
        ::onnxStatus status{::onnxInitEvent(backend, &event)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        status = ::onnxSignalEvent(event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        status = ::onnxSignalEvent(event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_STATE);
        status = ::onnxReleaseEvent(event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

// ====================================================[ onnxWaitEvent ]========

TEST(onnxifi, wait_event_invalid_event)
{
    ::onnxStatus status{::onnxWaitEvent(nullptr)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_EVENT);
}

TEST(onnxifi, wait_event)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxEvent input_event;
        ::onnxStatus status{::onnxInitEvent(backend, &input_event)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        ::onnxEvent output_event;
        status = ::onnxInitEvent(backend, &output_event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        std::size_t value{0};

        std::thread thread{[&]{
            ::onnxStatus local_status{::onnxWaitEvent(input_event)};
            if (local_status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{local_status};
            }
            value += 100;
            local_status = ::onnxSignalEvent(output_event);
            if (local_status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{local_status};
            }
        }};

        value += 100;
        status = ::onnxSignalEvent(input_event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        status = ::onnxWaitEvent(output_event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        EXPECT_TRUE(value == 200);
        thread.join();

        status = ::onnxReleaseEvent(input_event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        status = ::onnxReleaseEvent(output_event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

// ================================================[ onnxGetEventState ]=======

TEST(onnxifi, get_event_state_invalid_pointer)
{
    InitializedBackends backends{};
    for (const auto& backend : backends)
    {
        ::onnxEvent event;
        ::onnxStatus status{::onnxInitEvent(backend, &event)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        status = ::onnxGetEventState(event, nullptr);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);
        status = ::onnxReleaseEvent(event);
    }
}

TEST(onnxifi, get_event_state_invalid_event)
{
    ::onnxEventState state;
    ::onnxStatus status{::onnxGetEventState(nullptr, &state)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_EVENT);
}

TEST(onnxifi, get_event_state_nonsignaled)
{
    InitializedBackends backends{};
    for (const auto& backend : backends)
    {
        ::onnxEvent event;
        ::onnxStatus status{::onnxInitEvent(backend, &event)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        ::onnxEventState state;
        status = ::onnxGetEventState(event, &state);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        EXPECT_TRUE(state == ONNXIFI_EVENT_STATE_NONSIGNALLED);
        status = ::onnxReleaseEvent(event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, get_event_state_signaled)
{
    InitializedBackends backends{};
    for (const auto& backend : backends)
    {
        ::onnxEvent event;
        ::onnxStatus status{::onnxInitEvent(backend, &event)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        status = ::onnxSignalEvent(event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        ::onnxEventState state;
        status = ::onnxGetEventState(event, &state);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        EXPECT_TRUE(state == ONNXIFI_EVENT_STATE_SIGNALLED);
        status = ::onnxReleaseEvent(event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

// ====================================================[ onnxInitGraph ]=======

TEST(onnxifi, init_graph_invalid_pointer_weights)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph{(::onnxGraph)100};
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 100, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);
        EXPECT_TRUE(graph == nullptr);
    }
}

TEST(onnxifi, init_graph_invalid_backend)
{
    auto model = load_model();
    ::onnxGraph graph{(::onnxGraph)100};
    ::onnxStatus status{::onnxInitGraph(nullptr, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_BACKEND);
    EXPECT_TRUE(graph == nullptr);
}

TEST(onnxifi, init_graph_invalid_pointer_graph)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph{(::onnxGraph)100};
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, nullptr)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);
    }
}

TEST(onnxifi, init_graph_invalid_pointer_model)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph{(::onnxGraph)100};
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), nullptr, 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);
        EXPECT_TRUE(graph == nullptr);
    }
}

TEST(onnxifi, init_graph_invalid_size_model)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph{(::onnxGraph)100};
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, 0, model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_SIZE);
        EXPECT_TRUE(graph == nullptr);
    }
}

TEST(onnxifi, init_graph_invalid_size_weights)
{
    InitializedBackends backends{};
    auto model = load_model();
    ::onnxTensorDescriptorV1 weights;
    for (const auto& backend : backends)
    {
        ::onnxGraph graph{(::onnxGraph)100};
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, &weights, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_SIZE);
        EXPECT_TRUE(graph == nullptr);
    }
}

TEST(onnxifi, init_graph_invalid_protobuf)
{
    InitializedBackends backends{};
    std::string model{"invalid protobuf data"};

    for (const auto& backend : backends)
    {
        ::onnxGraph graph{(::onnxGraph)100};
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_PROTOBUF);
        EXPECT_TRUE(graph == nullptr);
    }
}

// =================================================[ onnxReleaseGraph ]=======


TEST(onnxifi, release_graph)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, release_graph_invalid_graph)
{
    ::onnxStatus status{::onnxReleaseGraph(nullptr)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_GRAPH);
}


// ==================================================[ onnxSetGraphIO ]========

namespace
{
    template <int32_t Tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1,
              ::onnxEnum Type = ONNXIFI_DATATYPE_FLOAT32,
              ::onnxEnum MemoryType = ONNXIFI_MEMORY_TYPE_CPU>
    struct TensorDescriptor_Template : ::onnxTensorDescriptorV1
    {
        TensorDescriptor_Template(const char* name, const void* data, uint32_t dimensions = 0, const uint64_t* shape = nullptr)
            : ::onnxTensorDescriptorV1{Tag, name, Type, MemoryType, dimensions, shape, reinterpret_cast<::onnxPointer>(data)}
        {
        }
    };

    using TensorDescriptor = TensorDescriptor_Template<>;
    using TensorDescriptor_InvalidTag = TensorDescriptor_Template<-1>;
    using TensorDescriptor_InvalidDataType = TensorDescriptor_Template<ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1, ONNXIFI_DATATYPE_COMPLEX128>;
    using TensorDescriptor_UnsupportedDataType = TensorDescriptor_Template<ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1, (::onnxEnum)-1>;
    using TensorDescriptor_InvalidMemoryType = TensorDescriptor_Template<ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1, ONNXIFI_DATATYPE_FLOAT32, ONNXIFI_MEMORY_TYPE_D3D_RESOURCE>;
    using TensorDescriptor_UnsupportedMemoryType = TensorDescriptor_Template<ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1, ONNXIFI_DATATYPE_FLOAT32, 128>;

} // namespace <anonymous>

TEST(onnxifi, set_graph_io_invalid_graph)
{
    ::onnxTensorDescriptorV1 input, output;
    ::onnxStatus status{::onnxSetGraphIO(nullptr, 1, &input, 1, &output)};
    EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_GRAPH);
}

TEST(onnxifi, set_graph_io_invalid_pointer)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float data{0.f};
        TensorDescriptor tensor{"A", &data};

        // It is allowed not to specify inputs
        status = ::onnxSetGraphIO(graph, 0, nullptr, 1, &tensor);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        status = ::onnxSetGraphIO(graph, 1, &tensor, 1, nullptr);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, set_graph_io_invalid_name)
{
    // TODO: implement the test for ONNXIFI_STATUS_INVALID_NAME
    //
    //       The function call failed because one of the names in tensor descriptors doesn't
    //       match blob name in ModelProto.graph.input or ModelProto.graph.output, or the
    //       same name appears in more than one tensor descriptor.
}


TEST(onnxifi, set_graph_io_invalid_shape)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float input_data{0.f}, output_data{0.f};
        std::uint64_t shape{0};
        TensorDescriptor invalid_shape{"S", &input_data, 1, &shape};
        TensorDescriptor input{"A", &input_data}, output{"C", &output_data};

        status = ::onnxSetGraphIO(graph, 1, &invalid_shape, 1, &output);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_SHAPE);

        status = ::onnxSetGraphIO(graph, 1, &input, 1, &invalid_shape);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_SHAPE);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, set_graph_io_invalid_datatype)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float input_data{0.f}, output_data{0.f};
        TensorDescriptor_InvalidDataType invalid_datatype{"A", &input_data};
        TensorDescriptor input{"A", &input_data}, output{"C", &output_data};

        status = ::onnxSetGraphIO(graph, 1, &invalid_datatype, 1, &output);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_DATATYPE);

        status = ::onnxSetGraphIO(graph, 1, &input, 1, &invalid_datatype);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_DATATYPE);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, set_graph_io_invalid_memory_type)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float input_data{0.f}, output_data{0.f};
        TensorDescriptor_InvalidMemoryType invalid_datatype{"S", &input_data};
        TensorDescriptor input{"A", &input_data}, output{"C", &output_data};

        status = ::onnxSetGraphIO(graph, 1, &invalid_datatype, 1, &output);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_MEMORY_TYPE);

        status = ::onnxSetGraphIO(graph, 1, &input, 1, &invalid_datatype);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_MEMORY_TYPE);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, set_graph_io_invalid_memory_location)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float input_data{0.f}, output_data{0.f};
        TensorDescriptor invalid_memory_location{"S", nullptr};
        TensorDescriptor input{"A", &input_data}, output{"C", &output_data};

        status = ::onnxSetGraphIO(graph, 1, &invalid_memory_location, 1, &output);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_MEMORY_LOCATION);

        status = ::onnxSetGraphIO(graph, 1, &input, 1, &invalid_memory_location);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_MEMORY_LOCATION);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, set_graph_io_unsupported_tag)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float input_data{0.f}, output_data{0.f};
        TensorDescriptor_InvalidTag invalid_tag{"S", &input_data};
        TensorDescriptor input{"A", &input_data}, output{"C", &output_data};

        status = ::onnxSetGraphIO(graph, 1, &invalid_tag, 1, &output);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_TAG);

        status = ::onnxSetGraphIO(graph, 1, &input, 1, &invalid_tag);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_TAG);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, set_graph_io_unsupported_shape)
{
    // TODO: Implement the test for ONNXIFI_STATUS_UNSUPPORTED_SHAPE
    //
    //       The function call failed because the backend does not support
    //       the tensor shape in an input or output of one of the operators.
    //       The problematic tensor shape could be directly specified through
    //       `inputDescriptors` or `outputDescriptors` argument, or inferred
    //       from the inputs and outputs, and the problematic tenosr shape
    //       was provided the ValueInfoProto as a symbolic variable.
}

TEST(onnxifi, set_graph_io_unsupported_memory_type)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float input_data{0.f}, output_data{0.f};
        TensorDescriptor_UnsupportedMemoryType unsupported_memory_type{"S", &input_data};
        TensorDescriptor input{"A", &input_data}, output{"C", &output_data};

        status = ::onnxSetGraphIO(graph, 1, &unsupported_memory_type, 1, &output);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_MEMORY_TYPE);

        status = ::onnxSetGraphIO(graph, 1, &input, 1, &unsupported_memory_type);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_MEMORY_TYPE);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }

}

TEST(onnxifi, set_graph_io_unsupported_datatype)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float input_data{0.f}, output_data{0.f};
        TensorDescriptor_UnsupportedDataType invalid_datatype{"A", &input_data};
        TensorDescriptor input{"A", &input_data}, output{"C", &output_data};

        status = ::onnxSetGraphIO(graph, 1, &invalid_datatype, 1, &output);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_DATATYPE);

        status = ::onnxSetGraphIO(graph, 1, &input, 1, &invalid_datatype);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_DATATYPE);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, set_graph_io_unidentified_name)
{
    // TODO: Implement the test for ONNXIFI_STATUS_UNIDENTIFIED_NAME
    //
    //       The function call failed because one of the ValueInfoProto.name value in
    //       ModelProto.graph.input or ModelProto.grapth.output doesn't have a match
    //       in the inputDescriptors or outputDescriptors.
}

TEST(onnxifi, set_graph_io_mismatching_shape)
{
    // TODO: Implement the test for ONNXIFI_STATUS_MISMATCHING_SHAPE
    //
    //       The function call failed because the shapes specified through
    //       inputDescriptors or outputDescriptions argument are inconsistent with the
    //       shapes specified in the ONNX model graph.
}

TEST(onnxifi, set_graph_io_mismatching_datatype)
{
    // TODO: Implement the test for ONNXIFI_STATUS_MISMATCHING_DATATYPE
    //
    //        The function call failed because data types specified through
    //        inputDescriptors or outputDescriptors argument are inconsistent with the
    //        data types specified in the ONNX model graph.
}

// ====================================================[ onnxRunGraph ]========

namespace
{
    template <bool InitializeEvent = true>
    struct MemoryFence_Template : ::onnxMemoryFenceV1
    {
        MemoryFence_Template(const MemoryFence_Template&) = delete;
        MemoryFence_Template& operator=(const MemoryFence_Template&) = delete;

        MemoryFence_Template() = delete;

        MemoryFence_Template(MemoryFence_Template&&) noexcept = default;
        MemoryFence_Template& operator=(MemoryFence_Template&&) noexcept = default;

        MemoryFence_Template(::onnxBackend backend, int32_t tag, ::onnxEnum type)
            : ::onnxMemoryFenceV1{tag, type, nullptr}
        {
            if (InitializeEvent)
            {
                ::onnxStatus status{::onnxInitEvent(backend, &event)};
                if (status != ONNXIFI_STATUS_SUCCESS)
                {
                    throw error::status{status};
                }
            }
        }

        explicit MemoryFence_Template(::onnxBackend backend)
            : MemoryFence_Template{backend, ONNXIFI_TAG_MEMORY_FENCE_V1, ONNXIFI_SYNCHRONIZATION_EVENT}
        {
        }

        MemoryFence_Template(::onnxBackend backend, int32_t tag)
            : MemoryFence_Template{backend, tag, ONNXIFI_SYNCHRONIZATION_EVENT}
        {
        }

        MemoryFence_Template(::onnxBackend backend, ::onnxEnum type)
            : MemoryFence_Template{backend, ONNXIFI_TAG_MEMORY_FENCE_V1, type}
        {
        }

        MemoryFence_Template(::onnxBackend backend, ::onnxEvent event)
            : ::onnxMemoryFenceV1{ONNXIFI_TAG_MEMORY_FENCE_V1, ONNXIFI_SYNCHRONIZATION_EVENT, event}
        {
        }

        ~MemoryFence_Template()
        {
            // Read the status code, but ignore it. Here we may get
            // invalid event handle status because the test might want to try some.
            ::onnxStatus status{::onnxReleaseEvent(event)};
        }
    };

    using MemoryFence = MemoryFence_Template<true>;
    using MemoryFence_OneShot = MemoryFence_Template<false>;

} // namespace  anonymous

TEST(onnxifi, run_graph_invalid_pointer)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        MemoryFence input_fence{backend}, output_fence{backend};

        status = ::onnxRunGraph(graph, &input_fence, nullptr);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);

        status = ::onnxRunGraph(graph, nullptr, &output_fence);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_POINTER);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, run_graph_invalid_graph)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        MemoryFence input_fence{backend}, output_fence{backend};
        ::onnxStatus status{::onnxRunGraph(nullptr, &input_fence, &output_fence)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_GRAPH);
    }
}

TEST(onnxifi, run_graph_invalid_fence_type)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        // According to specification type of memory synchronization primitive
        // can accept either ONNXIFI_SYNCHRONIZATION_EVENT or ONNXIFI_SYNCHRONIZATION_IMPLICIT.
        // ONNXIFI_SYNCHRONIZATION_IMPLICIT is not used by nGraph ONNXIFI backend.
        MemoryFence invalid_fence_type{backend, 0xFFFFFFFFFFFFFFFFUL};
        MemoryFence input_fence{backend}, output_fence{backend};

        status = ::onnxRunGraph(graph, &invalid_fence_type, &output_fence);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_FENCE_TYPE);
        status = ::onnxRunGraph(graph, &input_fence, &invalid_fence_type);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_FENCE_TYPE);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, run_graph_invalid_event)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        MemoryFence invalid_event{backend, nullptr};
        MemoryFence output_fence{backend};

        // We only check input_event, because in case of 'one shot' scenario
        // the 'output_fence' event may be nullptr, therefor it shall be
        // allocated by an ONNXIFI backend.

        status = ::onnxRunGraph(graph, &invalid_event, &output_fence);
        EXPECT_TRUE(status == ONNXIFI_STATUS_INVALID_EVENT);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, run_graph_invalid_tag)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        MemoryFence invalid_tag{backend, (int32_t)0};
        MemoryFence input_fence{backend}, output_fence{backend};

        status = ::onnxRunGraph(graph, &invalid_tag, &output_fence);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_TAG);

        status = ::onnxRunGraph(graph, &input_fence, &invalid_tag);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_TAG);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

TEST(onnxifi, run_graph_unsupported_fence_type)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        MemoryFence unsupported_fence_type{backend, (::onnxEnum)ONNXIFI_SYNCHRONIZATION_IMPLICIT};
        MemoryFence input_fence{backend}, output_fence{backend};

        status = ::onnxRunGraph(graph, &unsupported_fence_type, &output_fence);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_FENCE_TYPE);

        status = ::onnxRunGraph(graph, &input_fence, &unsupported_fence_type);
        EXPECT_TRUE(status == ONNXIFI_STATUS_UNSUPPORTED_FENCE_TYPE);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

// ======================================================[ functional ]========

namespace
{
    ::onnxTensorDescriptorV1 get_tensor_descriptor(const char *name, const void* data)
    {
        return {ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1, name, ONNXIFI_DATATYPE_FLOAT32,
                ONNXIFI_MEMORY_TYPE_CPU, 1, nullptr, reinterpret_cast<::onnxPointer>(data)};
    }

    namespace detail
    {
        void run(::onnxBackend backend, const std::vector<char>& model, const std::vector<TensorDescriptor>& inputs,
                       std::vector<TensorDescriptor>& outputs)
        {
            ::onnxGraph graph;
            ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }

            status = ::onnxSetGraphIO(graph, (uint32_t) inputs.size(), inputs.data(), (uint32_t) outputs.size(), outputs.data());
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }

            MemoryFence input_fence{backend}, output_fence{backend};

            status = ::onnxRunGraph(graph, &input_fence, &output_fence);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }

            status = ::onnxSignalEvent(input_fence.event);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }

            status = ::onnxWaitEvent(output_fence.event);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }

            status = ::onnxReleaseGraph(graph);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw error::status{status};
            }
        }
    }

    bool run_model(const std::string& name, const std::vector<float>& inputs, const std::vector<float>& expected_outputs)
    {
        InitializedBackends backends{};
        auto model = load_model(name);
        for (const auto& backend : backends)
        {
            std::vector<TensorDescriptor> input_descriptors;
            for (const auto& input : inputs)
            {
                input_descriptors.emplace_back("Input", &input);
            }
            std::vector<TensorDescriptor> output_descriptors;
            std::vector<float> outputs(expected_outputs.size());
            for (auto& output : outputs)
            {
                output_descriptors.emplace_back("Output", &output);
            }

            detail::run(backend, model, input_descriptors, output_descriptors);
            if (!ngraph::test::all_close_f(expected_outputs, outputs))
            {
                return false;
            }
        }
        return true;
    }

    using Tensors = std::vector<std::vector<float>>;
    using Shapes = std::vector<std::size_t>;

    bool run_model(const std::string& name, const Tensors& inputs, const Shapes& input_shapes,
                   const Tensors& expected_outputs, const Shapes& expected_shapes)
    {
        InitializedBackends backends{};
        auto model = load_model(name);
        for (const auto& backend : backends)
        {
            std::vector<TensorDescriptor> input_descriptors;
            for (std::size_t i{0}; i < inputs.size(); ++i)
            {
                input_descriptors.emplace_back("Input", inputs[i].data(), input_shapes.size(), input_shapes.data());
            }
            std::vector<TensorDescriptor> output_descriptors;
            std::vector<std::vector<float>> outputs(expected_outputs.size());
            for (std::size_t i{0}; i < outputs.size(); ++i)
            {
                outputs[i].resize(expected_outputs[i].size(), -1.f);
                output_descriptors.emplace_back("Output", outputs[i].data(), expected_shapes.size(), expected_shapes.data());
            }

            detail::run(backend, model, input_descriptors, output_descriptors);
            for (std::size_t i{0}; i < expected_outputs.size(); ++i)
            {
                if (!ngraph::test::all_close_f(expected_outputs[i], outputs[i]))
                {
                    return false;
                }
            }
        }
        return true;
    }

    bool run_model(const std::string& name, const Tensors& inputs,
                   const Tensors& expected_outputs, const Shapes& shapes)
    {
        return run_model(name, inputs, shapes, expected_outputs, shapes);
    }

} // namespace <anonymous>

// If this test hangs it means onnxRunGraph() is not asynchronous.
// THE FORM OF THIS TEST IS BY PURPOSE: it checks setting input parameters after call to onnxRunGraph
TEST(onnxifi, model_add_abc)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float values[] = {0.f, 0.f, 0.f};
        std::vector<TensorDescriptor> inputs{
            {"A", &values[0]}, {"B", &values[1]}, {"C", &values[2]}
        };

        float output_placeholder{-1};
        TensorDescriptor output{"Result", &output_placeholder};

        status = ::onnxSetGraphIO(graph, (uint32_t)inputs.size(), inputs.data(), 1, &output);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        MemoryFence input_fence{backend}, output_fence{backend};

        status = ::onnxRunGraph(graph, &input_fence, &output_fence);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        values[0] = 1.f;
        values[1] = 2.f;
        values[2] = 3.f;

        status = ::onnxSignalEvent(input_fence.event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        status = ::onnxWaitEvent(output_fence.event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        EXPECT_TRUE(output_placeholder == 6.f);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

// If this test hangs it means onnxRunGraph() is not asynchronous.
// THE FORM OF THIS TEST IS BY PURPOSE: it checks setting input parameters after call to onnxRunGraph,
//                                      and the output fence event created by the ONNXIFI backend.
TEST(onnxifi, model_add_abc_oneshot)
{
    InitializedBackends backends{};
    auto model = load_model();
    for (const auto& backend : backends)
    {
        ::onnxGraph graph;
        ::onnxStatus status{::onnxInitGraph(backend, nullptr, model.size(), model.data(), 0, nullptr, &graph)};
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        float values[] = {0.f, 0.f, 0.f};
        std::vector<TensorDescriptor> inputs{
            {"A", &values[0]}, {"B", &values[1]}, {"C", &values[2]}
        };

        float output_placeholder{-1};
        TensorDescriptor output{"Result", &output_placeholder};

        status = ::onnxSetGraphIO(graph, (uint32_t)inputs.size(), inputs.data(), 1, &output);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        MemoryFence input_fence{backend};
        MemoryFence_OneShot output_fence{backend};

        status = ::onnxRunGraph(graph, &input_fence, &output_fence);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        values[0] = 1.f;
        values[1] = 2.f;
        values[2] = 3.f;

        status = ::onnxSignalEvent(input_fence.event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        status = ::onnxWaitEvent(output_fence.event);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);

        EXPECT_TRUE(output_placeholder == 6.f);

        status = ::onnxReleaseGraph(graph);
        EXPECT_TRUE(status == ONNXIFI_STATUS_SUCCESS);
    }
}

using namespace ngraph::test;

TEST(onnxifi, model_add_abc_initializers)
{
    EXPECT_TRUE(run_model("add_abc_initializers.onnx",
        {{1, 2, 3, 4}}, {{3, 6, 9, 12}}, {2, 2}));
}


TEST(onnxifi, model_addmul_abc)
{
    Tensors inputs{
        NDArray<float,3>{{{{9, 10}}, {{11, 12}}}}.get_vector(),
        NDArray<float,3>{{{{5, 6}}, {{7, 8}}}}.get_vector(),
        NDArray<float,3>{{{{1, 2}}, {{3, 4}}}}.get_vector()};

    Tensors outputs{
        NDArray<float,3>{{{{46, 62}}, {{80, 100}}}}.get_vector()};

    EXPECT_TRUE(run_model("addmul_abc.onnx", inputs, outputs, {1, 2, 2}));
}

TEST(onnxifi, model_average_pool_2d)
{
    Tensors inputs{
        NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f},
                             {4.f, 5.f, 6.f, 7.f},
                             {8.f, 9.f, 10.f, 11.f},
                             {12.f, 13.f, 14.f, 15.f}}}}}.get_vector()
    };
    Tensors outputs{
        NDArray<float, 4>({{{{2.5f, 4.5f}, {10.5f, 12.5f}}}}).get_vector()
    };
    EXPECT_TRUE(run_model("average_pool_2d.onnx", inputs, {1, 1, 4, 4}, outputs, {1, 1, 2, 2}));
}

TEST(onnxifi, model_average_pool_2d_pads)
{
    Tensors inputs{
        NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f},
                             {4.f, 5.f, 6.f, 7.f},
                             {8.f, 9.f, 10.f, 11.f},
                             {12.f, 13.f, 14.f, 15.f}}}}}.get_vector()
    };
    Tensors outputs{
        NDArray<float, 4>{{{{{0.f, 1.5f, 3.f}, {6.f, 7.5f, 9.f}, {12.f, 13.5f, 15.f}}}}}.get_vector()
    };
    EXPECT_TRUE(run_model("average_pool_2d_pads.onnx", inputs, {1, 1, 4, 4}, outputs, {1, 1, 3, 3}));
}

TEST(onnxifi, model_concat)
{
    Tensors inputs{
        NDArray<float, 1>{{1, 2}}.get_vector(),
        NDArray<float, 1>{{3, 4}}.get_vector()
    };
    Tensors outputs{
        NDArray<float, 1>{{1, 2, 3, 4}}.get_vector()
    };
    EXPECT_TRUE(run_model("concat.onnx", inputs, {2}, outputs, {4}));
}
