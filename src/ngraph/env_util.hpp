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

namespace ngraph
{
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

    /// \brief Adds the environment variable with it's value to a map
    /// \param env_var The string name of the environment variable to add.
    /// \param val The string value of the environment variable to add.
    void addenv_to_map(const char* env_var, const char* val);

    /// \brief Gets the environment variable with it's value to a map
    /// \param env_var The string name of the environment variable to add.
    /// \return Returns the boolean value of the environment variable.
    std::string getenv_from_map(const char* env_var);

    /// \brief Get the names environment variable as a string.
    /// \param env_var The string name of the environment variable to get.
    /// \return Returns string by value or an empty string if the environment
    ///         variable is not set.
    void log_all_envvar();

    /// \brief Set the environment variable.
    /// \param env_var The string name of the environment variable to set.
    /// \param val The string value of the environment variable to set.
    /// \param overwrite Flag to overwrite already set environment variable.
    ///         0 = do not overwrite.
    ///         1 = overwrite the environment variable with this new value.
    /// \return Returns 0 if successful, -1 in case of error.
    NGRAPH_API int set_environment(const char* env_var, const char* value, const int overwrite = 0);

    /// \brief Unset the environment variable.
    /// \param env_var The string name of the environment variable to unset.
    /// \return Returns 0 if successful, -1 in case of error.
    NGRAPH_API int unset_environment(const char* env_var);

    /// \brief Check if the environment variable is present in the cache map.
    /// \param env_var The string name of the environment variable to check.
    /// \return Returns true if found, else false.
    bool map_contains(const char* env_var);

    /// \brief Delete the environment variable from the cache map.
    /// \param env_var The string name of the environment variable to delete.
    void erase_env_from_map(const char* env_var);
}
