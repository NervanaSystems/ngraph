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

struct EnvVarInfo
{
    string env_str;
    string default_val;
    string desc;
};

std::unordered_map<ngraph::EnvVarEnum, std::string>& get_env_var_cache()
{
    static std::unordered_map<ngraph::EnvVarEnum, string> s_env_var_cache;
    return s_env_var_cache;
}

void addenv_to_cache(const ngraph::EnvVarEnum env_var, const char* val)
{
    get_env_var_cache().emplace(env_var, val);
}

bool env_cache_contains(const ngraph::EnvVarEnum env_var)
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

std::string getenv_from_cache(const ngraph::EnvVarEnum env_var)
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

void erase_env_from_cache(const ngraph::EnvVarEnum env_var)
{
    get_env_var_cache().erase(env_var);
}

void ngraph::log_envvar_cache()
{
    NGRAPH_DEBUG << "List of all environment variables:\n";
    std::unordered_map<ngraph::EnvVarEnum, std::string>::iterator it = get_env_var_cache().begin();
    while (it != get_env_var_cache().end())
    {
        NGRAPH_DEBUG << "\t" << get_env_var_name(it->first) << " = " << it->second << std::endl;
        it++;
    }
}

static map<ngraph::EnvVarEnum, EnvVarInfo> get_env_registry()
{
    // This expands the env var list in env_tbl.hpp into a list of enumerations that look like this:
    // {ngraph::EnvVarEnum::NGRAPH_CODEGEN, {"NGRAPH_CODEGEN", "FALSE", "Enable ngraph codegen"}},
    // {ngraph::EnvVarEnum::NGRAPH_COMPILER_DEBUGINFO_ENABLE, "NGRAPH_COMPILER_DEBUGINFO_ENABLE",
    //                            "FALSE", "Enable compiler debug info when codegen is enabled"}},
    // ...

    static const map<ngraph::EnvVarEnum, EnvVarInfo> envvar_info_map{
#define NGRAPH_DEFINE_ENVVAR(ENUMID, NAME, DEFAULT, DESCRIPTION)                                   \
    {ENUMID, {NAME, DEFAULT, DESCRIPTION}},
#include "ngraph/env_tbl.hpp"
#undef NGRAPH_DEFINE_ENVVAR
    };
    return envvar_info_map;
}

string ngraph::get_env_var_name(const ngraph::EnvVarEnum env_var)
{
    return get_env_registry()[env_var].env_str;
}

string ngraph::get_env_var_default(const ngraph::EnvVarEnum env_var)
{
    return get_env_registry()[env_var].default_val;
}

string ngraph::get_env_var_desc(const ngraph::EnvVarEnum env_var)
{
    return get_env_registry()[env_var].desc;
}

void ngraph::log_envvar_registry()
{
    NGRAPH_DEBUG << "List of all environment variables in registry:\n";
    for (uint32_t i = 0; i < uint32_t(ngraph::EnvVarEnum::NGRAPH_ENV_VARS_COUNT); i++)
    {
        NGRAPH_DEBUG << "\tEnum = " << i
                     << ", name = " << get_env_var_name(static_cast<ngraph::EnvVarEnum>(i))
                     << ", default = " << get_env_var_default(static_cast<ngraph::EnvVarEnum>(i))
                     << std::endl;
    }
}

int ngraph::set_environment(const ngraph::EnvVarEnum env_var_enum,
                            const char* value,
                            const int overwrite)
{
    const char* env_var = get_env_var_name(env_var_enum).c_str();
    if (env_cache_contains(env_var_enum) && !overwrite)
    {
        NGRAPH_WARN << "Cannot set environment variable " << env_var << " is already set to "
                    << value << ", and overwrite is false";
        return -1;
    }
    else if (env_cache_contains(env_var_enum))
    {
        erase_env_from_cache(env_var_enum);
    }
    addenv_to_cache(env_var_enum, value);

#ifdef _WIN32
    return _putenv_s(env_var, value);
#elif defined(__linux) || defined(__APPLE__)
    return setenv(env_var, value, overwrite);
#endif
}

int ngraph::unset_environment(const ngraph::EnvVarEnum env_var_enum)
{
    const char* env_var = get_env_var_name(env_var_enum).c_str();
    erase_env_from_cache(env_var_enum);
#ifdef _WIN32
    return _putenv_s(env_var, "");
#elif defined(__linux) || defined(__APPLE__)
    return unsetenv(env_var);
#endif
}

std::string ngraph::getenv_string(const ngraph::EnvVarEnum env_var)
{
    if (env_cache_contains(env_var))
    {
        string env_string = getenv_from_cache(env_var);
        return env_string;
    }
    else
    {
        const char* env_p = ::getenv(get_env_var_name(env_var).c_str());
        string env_string = env_p ? env_p : "";
        addenv_to_cache(env_var, env_string.c_str());
        return env_string;
    }
}

int32_t ngraph::getenv_int(const ngraph::EnvVarEnum env_var)
{
    char* err;
    if (env_cache_contains(env_var))
    {
        string env_p = getenv_from_cache(env_var);

        errno = 0;
        int32_t env_int = strtol(env_p.c_str(), &err, 0);
        if (errno == 0 || *err)
        {
            // Extensive error checking was done when reading getenv, keeping it minimal here, ok?
            NGRAPH_DEBUG << "Error reading (" << get_env_var_name(env_var)
                         << ") empty or undefined, "
                         << " defaulted to -1 here.";
            env_int = strtol(get_env_var_default(env_var).c_str(), &err, 0);
        }
        return env_int;
    }
    else
    {
        const char* env_p = ::getenv(get_env_var_name(env_var).c_str());
        int32_t env_int = strtol(get_env_var_default(env_var).c_str(), &err, 0);
        // If env_var is not "" or undefined
        if (env_p && *env_p)
        {
            errno = 0;
            char* err;
            env_int = strtol(env_p, &err, 0);
            // if conversion leads to an overflow
            if (errno)
            {
                std::stringstream ss;
                ss << "Environment variable \"" << get_env_var_name(env_var).c_str() << "\"=\""
                   << env_p << "\" converted to different value \"" << env_int
                   << "\" due to overflow." << std::endl;
                throw runtime_error(ss.str());
            }
            // if syntax error is there - conversion will still happen
            // but warn user of syntax error
            if (*err)
            {
                std::stringstream ss;
                ss << "Environment variable \"" << get_env_var_name(env_var).c_str() << "\"=\""
                   << env_p << "\" converted to different value \"" << env_int
                   << "\" due to syntax error \"" << err << '\"' << std::endl;
                throw runtime_error(ss.str());
            }
        }
        else
        {
            NGRAPH_DEBUG << "Environment variable (" << env_var << ") empty or undefined, "
                         << " defaulted to -1 here.";
        }
        addenv_to_cache(env_var, std::to_string(env_int).c_str());
        return env_int;
    }
}

bool ngraph::getenv_bool(const ngraph::EnvVarEnum env_var)
{
    string value = to_lower(getenv_string(env_var));
    static const set<string> off = {"0", "false", "off", "FALSE", "OFF", "no", "NO"};
    static const set<string> on = {"1", "true", "on", "TRUE", "ON", "yes", "YES"};
    bool rc = false;
    if (value == "")
    {
        rc = false;
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
        ss << "environment variable '" << get_env_var_name(env_var).c_str() << "' value '" << value
           << "' invalid. Must be boolean.";
        throw runtime_error(ss.str());
    }
    return rc;
}
