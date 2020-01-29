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
    NGRAPH_API enum class EnvVarEnum : uint32_t
    {
        NGRAPH_CODEGEN = 0,
        NGRAPH_COMPILER_DEBUGINFO_ENABLE = 1,
        NGRAPH_COMPILER_DIAG_ENABLE = 2,
        NGRAPH_COMPILER_REPORT_ENABLE = 3,
        NGRAPH_CPU_BIN_TRACER_LOG = 4,
        NGRAPH_CPU_CHECK_PARMS_AND_CONSTS = 5,
        NGRAPH_CPU_CONCURRENCY = 6,
        NGRAPH_CPU_DEBUG_TRACER = 7,
        NGRAPH_CPU_EIGEN_THREAD_COUNT = 8,
        NGRAPH_CPU_INF_CHECK = 9,
        NGRAPH_CPU_NAN_CHECK = 10,
        NGRAPH_CPU_TRACER_LOG = 11,
        NGRAPH_CPU_TRACING = 12,
        NGRAPH_CPU_USE_REF_KERNELS = 13,
        NGRAPH_CPU_USE_TBB = 14,
        NGRAPH_DECONV_FUSE = 15,
        NGRAPH_DEX_DEBUG = 16,
        NGRAPH_DISABLE_LOGGING = 17,
        NGRAPH_DISABLED_FUSIONS = 18,
        NGRAPH_ENABLE_REPLACE_CHECK = 19,
        NGRAPH_ENABLE_SERIALIZE_TRACING = 20,
        NGRAPH_ENABLE_TRACING = 21,
        NGRAPH_ENABLE_VISUALIZE_TRACING = 22,
        NGRAPH_FAIL_MATCH_AT = 23,
        NGRAPH_GRAPH_REWRITE_RERUN_DYNAMIC_CHECK = 24,
        NGRAPH_GTEST_INFO = 25,
        NGRAPH_INTER_OP_PARALLELISM = 26,
        NGRAPH_INTRA_OP_PARALLELISM = 27,
        NGRAPH_MLIR = 28,
        NGRAPH_MLIR_MAX_CYCLE_DEPTH = 29,
        NGRAPH_MLIR_OPT_LEVEL = 30,
        NGRAPH_MLIR_OPTIONS = 31,
        NGRAPH_PASS_ATTRIBUTES = 32,
        NGRAPH_PASS_CPU_LAYOUT_ELTWISE = 33,
        NGRAPH_PASS_ENABLES = 34,
        NGRAPH_PROFILE_PASS_ENABLE = 35,
        NGRAPH_PROVENANCE_ENABLE = 36,
        NGRAPH_SERIALIZER_OUTPUT_SHAPES = 37,
        NGRAPH_VISUALIZE_EDGE_JUMP_DISTANCE = 38,
        NGRAPH_VISUALIZE_EDGE_LABELS = 39,
        NGRAPH_VISUALIZE_TRACING_FORMAT = 40,
        NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES = 41,
        NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES = 42,
        OMP_NUM_THREADS = 43,
        NGRAPH_TMPDIR = 44,
        NGRAPH_ENV_VARS_COUNT
    };

    /// \brief Get the names environment variable as a string.
    /// \param env_var The enum value of the environment variable to get.
    /// \return Returns string by value or an empty string if the environment
    ///         variable is not set.
    NGRAPH_API std::string getenv_string(const EnvVarEnum env_var);

    /// \brief Get the names environment variable as an integer. If the value is not a
    ///        valid integer then an exception is thrown.
    /// \param env_var enum value of the environment variable to get.
    /// \param default_value The value to return if the environment variable is not set.
    /// \return Returns value or default_value if the environment variable is not set.
    NGRAPH_API int32_t getenv_int(const EnvVarEnum env_var);

    /// \brief Get the names environment variable as a boolean. If the value is not a
    ///        valid boolean then an exception is thrown. Valid booleans are one of
    ///        1, 0, on, off, true, false
    ///        All values are case insensitive.
    ///        If the environment variable is not set the default_value is returned.
    /// \param env_var The enum value of the environment variable to get.
    /// \param default_value The value to return if the environment variable is not set.
    /// \return Returns the boolean value of the environment variable.
    NGRAPH_API bool getenv_bool(const EnvVarEnum env_var);

    /// \brief Logs the current environment variables and their current values.
    NGRAPH_API void log_envvar_cache();

    /// \brief Logs all environment variables and their default values.
    NGRAPH_API void log_envvar_registry();

    /// \brief Set the environment variable.
    /// \param env_var The enum value of the environment variable to set.
    /// \param val The enum value of the environment variable to set.
    /// \param overwrite Flag to overwrite already set environment variable.
    ///         0 = do not overwrite.
    ///         1 = overwrite the environment variable with this new value.
    /// \return Returns 0 if successful, -1 in case of error.
    NGRAPH_API int
        set_environment(const EnvVarEnum env_var, const char* value, const int overwrite = 0);

    /// \brief Unset the environment variable.
    /// \param env_var The enum value of the environment variable to unset.
    /// \return Returns 0 if successful, -1 in case of error.
    NGRAPH_API int unset_environment(const EnvVarEnum env_var);

    /// \brief Get string name of the environment variable from the registry.
    /// \param env_var The enum value of the environment variable.
    NGRAPH_API string get_env_var_name(const ngraph::EnvVarEnum env_var);

    /// \brief Get default value of the environment variable from the registry.
    /// \param env_var The enum value of the environment variable.
    NGRAPH_API string get_env_var_default(const ngraph::EnvVarEnum env_var);

    /// \brief Get description of the environment variable from the registry.
    /// \param env_var The enum value of the environment variable.
    NGRAPH_API string get_env_var_desc(const ngraph::EnvVarEnum env_var);
}
