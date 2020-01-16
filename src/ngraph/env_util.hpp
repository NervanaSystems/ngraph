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

#pragma once

#include <string>
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/util.hpp"

using namespace std;

namespace ngraph
{
    /// \brief List of currently supported environment variables
    enum class EnvVarEnum : uint32_t
    {
        NGRAPH_CODEGEN = 0,
        NGRAPH_COMPILER_DEBUGINFO_ENABLE=1,
        NGRAPH_COMPILER_DIAG_ENABLE=2,
        NGRAPH_COMPILER_REPORT_ENABLE=3,
        NGRAPH_CPU_BIN_TRACER_LOG=4,
        NGRAPH_CPU_CHECK_PARMS_AND_CONSTS=5,
        NGRAPH_CPU_CONCURRENCY=6,
        NGRAPH_CPU_DEBUG_TRACER=7,
        NGRAPH_CPU_EIGEN_THREAD_COUNT=8,
        NGRAPH_CPU_INF_CHECK=9,
        NGRAPH_CPU_NAN_CHECK=10,
        NGRAPH_CPU_TRACER_LOG=11,
        NGRAPH_CPU_TRACING=12,
        NGRAPH_CPU_USE_REF_KERNELS=13,
        NGRAPH_CPU_USE_TBB=14,
        NGRAPH_DECONV_FUSE=15,
        NGRAPH_DEX_DEBUG=16,
        NGRAPH_DISABLE_LOGGING=17,
        NGRAPH_DISABLED_FUSIONS=18,
        NGRAPH_ENABLE_REPLACE_CHECK=19,
        NGRAPH_ENABLE_SERIALIZE_TRACING=20,
        NGRAPH_ENABLE_TRACING=21,
        NGRAPH_ENABLE_VISUALIZE_TRACING=22,
        NGRAPH_FAIL_MATCH_AT=23,
        NGRAPH_GRAPH_REWRITE_RERUN_DYNAMIC_CHECK=24,
        NGRAPH_GTEST_INFO=25,
        NGRAPH_INTER_OP_PARALLELISM=26,
        NGRAPH_INTRA_OP_PARALLELISM=27,
        NGRAPH_MLIR=28,
        NGRAPH_MLIR_MAX_CYCLE_DEPTH=29,
        NGRAPH_MLIR_OPT_LEVEL=30,
        NGRAPH_MLIR_OPTIONS=31,
        NGRAPH_PASS_ATTRIBUTES=32,
        NGRAPH_PASS_CPU_LAYOUT_ELTWISE=33,
        NGRAPH_PASS_ENABLES=34,
        NGRAPH_PROFILE_PASS_ENABLE=35,
        NGRAPH_PROVENANCE_ENABLE=36,
        NGRAPH_SERIALIZER_OUTPUT_SHAPES=37,
        NGRAPH_VISUALIZE_EDGE_JUMP_DISTANCE=38,
        NGRAPH_VISUALIZE_EDGE_LABELS=39,
        NGRAPH_VISUALIZE_TRACING_FORMAT=40,
        NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES=41,
        NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES=42,
        OMP_NUM_THREADS=43,
        NGRAPH_MAX_ENV_VAR=44
    };
}
template class NGRAPH_API ngraph::EnumMask<ngraph::EnvVarEnum>;

namespace ngraph
{
    typedef EnumMask<EnvVarEnum> EnvVarEnumMask;

    struct EnvVarInfo
    {
        string env_str;
        string default_val;
        string desc;
    };

    //--------------------------------------------------- temp line to 


    /// \brief Get the names environment variable as a string.
    /// \param env_var The string name of the environment variable to get.
    /// \return Returns string by value or an empty string if the environment
    ///         variable is not set.
    std::string getenv_string(const char* env_var);

    /// \brief Get the names environment variable as an integer. If the value is not a
    ///        valid integer then an exception is thrown.
    /// \param env_var The string name of the environment variable to get.
    /// \param default_value The value to return if the environment variable is not set.
    /// \return Returns value or default_value if the environment variable is not set.
    int32_t getenv_int(const char* env_var, int32_t default_value = -1);

    /// \brief Get the names environment variable as a boolean. If the value is not a
    ///        valid boolean then an exception is thrown. Valid booleans are one of
    ///        1, 0, on, off, true, false
    ///        All values are case insensitive.
    ///        If the environment variable is not set the default_value is returned.
    /// \param env_var The string name of the environment variable to get.
    /// \param default_value The value to return if the environment variable is not set.
    /// \return Returns the boolean value of the environment variable.
    bool getenv_bool(const char* env_var, bool default_value = false);

    /// \brief Adds the environment variable with it's value to the map.
    /// \param env_var The string name of the environment variable to add.
    /// \param val The string value of the environment variable to add.
    void addenv_to_map(const char* env_var, const char* val);

    /// \brief Gets value of the environment variable from the map.
    /// \param env_var The string name of the environment variable to get.
    /// \return Returns string value of the environment variable.
    std::string getenv_from_map(const char* env_var);

    /// \brief Logs the current environment variables and their values.
    void log_all_envvar();

    /// \brief Set the environment variable.
    /// \param env_var The string name of the environment variable to set.
    /// \param val The string value of the environment variable to set.
    /// \param overwrite Flag to overwrite already set environment variable.
    ///         0 = do not overwrite.
    ///         1 = overwrite the environment variable with this new value.
    /// \return Returns 0 if successful, -1 in case of error.
    //template <typename ET>
    //NGRAPH_API int set_environment(ET env_var, const char* value, const int overwrite = 0);
    NGRAPH_API int set_environment(EnvVarEnumMask env_var, const char* value, const int overwrite = 0);

    /// \brief Unset the environment variable.
    /// \param env_var The string name of the environment variable to unset.
    /// \return Returns 0 if successful, -1 in case of error.
    //template <typename ET>
    NGRAPH_API int unset_environment(EnvVarEnumMask env_var);

    /// \brief Check if the environment variable is present in the cache map.
    /// \param env_var The string name of the environment variable to check.
    /// \return Returns true if found, else false.
    bool map_contains(const char* env_var);

    /// \brief Delete the environment variable from the cache map.
    /// \param env_var The string name of the environment variable to delete.
    void erase_env_from_map(const char* env_var);
}
