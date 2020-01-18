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

#include <sstream>

#include <unordered_map>
#include <utility>
#include "ngraph/env_util.hpp"
#include "ngraph/util.hpp"

using namespace std;

static map<ngraph::EnvVarEnum, ngraph::EnvVarInfo> get_env_registry()
{
    // This expands the env var list in env_tbl.hpp into a list of enumerations that look like this:
    // {ngraph::EnvVarEnum::NGRAPH_CODEGEN, {"NGRAPH_CODEGEN", "FALSE", "Enable ngraph codegen"}},
    // {ngraph::EnvVarEnum::NGRAPH_COMPILER_DEBUGINFO_ENABLE, "NGRAPH_COMPILER_DEBUGINFO_ENABLE",
    //                            "FALSE", "Enable compiler debug info when codegen is enabled"}},
    // ...
    static const map<ngraph::EnvVarEnum, ngraph::EnvVarInfo> envvar_info_map{
#define NGRAPH_DEFINE_ENVVAR(ENUMID, NAME, DEFAULT, DESCRIPTION)                                   \
    {ENUMID, {NAME, DEFAULT, DESCRIPTION}},
#include "ngraph/env_tbl.hpp"
#undef NGRAPH_DEFINE_ENVVAR
    };
    return envvar_info_map;
}

// get current value or default
static ngraph::EnvVarInfo& get_env_registry_info(const ngraph::EnvVarEnum env_var)
{
    if (env_var > ngraph::EnvVarEnum::NGRAPH_MAX_ENV_VAR)
    {
        throw "Unknown environment variable enum. Should not happen\n";
    }
    auto it = get_env_registry().find(env_var);
    return it->second;
}

static string get_env_registry_name(const ngraph::EnvVarEnum env_var)
{
    return get_env_registry_info(env_var).env_str;
}

static string get_env_var_default(const ngraph::EnvVarEnum env_var)
{
    return get_env_registry_info(env_var).default_val;
}

static string get_env_var_desc(const ngraph::EnvVarEnum env_var)
{
    return get_env_registry_info(env_var).desc;
}

// Above this is registry related stuff
//--------------------------

// template <typename ET>
int ngraph::set_environment(EnvVarEnum env_var_enum, const char* value, const int overwrite)
{
    const char* env_var = get_env_registry_name(env_var_enum).c_str();
    if (env_cache_contains(env_var) && !overwrite)
    {
        NGRAPH_WARN << "Cannot set environment variable " << env_var << " is already set to "
                    << value << ", and overwrite is false";
        return -1;
    }
    else if (env_cache_contains(env_var))
    {
        erase_env_from_cache(env_var);
    }
    addenv_to_cache(env_var, value);

#ifdef _WIN32
    return _putenv_s(env_var, value);
#elif defined(__linux) || defined(__APPLE__)
    return setenv(env_var, value, overwrite);
#endif
}

// template <typename ET>
int ngraph::unset_environment(EnvVarEnum env_var_enum)
{
    const char* env_var = get_env_registry_name(env_var_enum).c_str();
    erase_env_from_cache(env_var);
#ifdef _WIN32
    return _putenv_s(env_var, "");
#elif defined(__linux) || defined(__APPLE__)
    return unsetenv(env_var);
#endif
}
// set and unset programmatical apis
//----------------------

// --------- below this is caching apis
std::unordered_map<std::string, std::string>& get_env_var_cache()
{
    static std::unordered_map<string, string> s_env_var_cache;
    return s_env_var_cache;
}

void ngraph::log_all_envvar()
{
    NGRAPH_DEBUG << "List of all environment variables:\n";
    std::unordered_map<std::string, std::string>::iterator it = get_env_var_cache().begin();
    while (it != get_env_var_cache().end())
    {
        NGRAPH_DEBUG << "\t" << it->first << " = " << it->second << std::endl;
        it++;
    }
}

void ngraph::addenv_to_cache(const char* env_var, const char* val)
{
    get_env_var_cache().emplace(env_var, val);
}

bool ngraph::env_cache_contains(const char* env_var)
{
    if (get_env_var_cache().find(env_var) != get_env_var_cache().end())
    {
        return true;
    }
    else
    {
        return false;
    }
}

std::string ngraph::getenv_from_cache(const char* env_var)
{
    if (env_cache_contains(env_var))
    {
        return get_env_var_cache().at(env_var);
    }
    else
    {
        return "";
    }
}

void ngraph::erase_env_from_cache(const char* env_var)
{
    get_env_var_cache().erase(env_var);
}

std::string ngraph::getenv_string(const char* env_var)
{
    if (env_cache_contains(env_var))
    {
        return getenv_from_cache(env_var);
    }
    else
    {
        const char* env_p = ::getenv(env_var);
        string env_string = env_p ? env_p : "";
        addenv_to_cache(env_var, env_string.c_str());
        return env_string;
    }
}

int32_t ngraph::getenv_int(const char* env_var, int32_t default_value)
{
    if (env_cache_contains(env_var))
    {
        string env_p = getenv_from_cache(env_var);

        errno = 0;
        char* err;
        int32_t env_int = strtol(env_p.c_str(), &err, 0);
        if (errno == 0 || *err)
        {
            // Extensive error checking was done when reading getenv, keeping it minimal here, ok?
            NGRAPH_DEBUG << "Error reading (" << env_var << ") empty or undefined, "
                         << " defaulted to -1 here.";
            return default_value;
        }
        return env_int;
    }
    else
    {
        const char* env_p = ::getenv(env_var);
        int32_t env = default_value;
        // If env_var is not "" or undefined
        if (env_p && *env_p)
        {
            errno = 0;
            char* err;
            env = strtol(env_p, &err, 0);
            // if conversion leads to an overflow
            if (errno)
            {
                std::stringstream ss;
                ss << "Environment variable \"" << env_var << "\"=\"" << env_p
                   << "\" converted to different value \"" << env << "\" due to overflow."
                   << std::endl;
                throw runtime_error(ss.str());
            }
            // if syntax error is there - conversion will still happen
            // but warn user of syntax error
            if (*err)
            {
                std::stringstream ss;
                ss << "Environment variable \"" << env_var << "\"=\"" << env_p
                   << "\" converted to different value \"" << env << "\" due to syntax error \""
                   << err << '\"' << std::endl;
                throw runtime_error(ss.str());
            }
        }
        else
        {
            NGRAPH_DEBUG << "Environment variable (" << env_var << ") empty or undefined, "
                         << " defaulted to -1 here.";
        }
        addenv_to_cache(env_var, std::to_string(env).c_str());
        return env;
    }
}

bool ngraph::getenv_bool(const char* env_var, bool default_value)
{
    string value = to_lower(getenv_string(env_var));
    set<string> off = {"0", "false", "off"};
    set<string> on = {"1", "true", "on"};
    bool rc;
    if (value == "")
    {
        rc = default_value;
    }
    else if (off.find(value) != off.end())
    {
        rc = false;
    }
    else if (on.find(value) != on.end())
    {
        rc = true;
    }
    else
    {
        stringstream ss;
        ss << "environment variable '" << env_var << "' value '" << value
           << "' invalid. Must be boolean.";
        throw runtime_error(ss.str());
    }
    return rc;
}
