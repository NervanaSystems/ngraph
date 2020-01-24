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


/*static ngraph::EnvVarInfo& get_env_registry_info(const ngraph::EnvVarEnum env_var)
{
    if (env_var > ngraph::EnvVarEnum::NGRAPH_MAX_ENV_VAR)
    {
        throw "Unknown environment variable enum. Should not happen\n";
    }
    auto it = get_env_registry().find(env_var);
    //std::cout << "\n\nget_env_registry_info it-> second  name = " << it->second.env_str << std::endl;
    std::cout << "\n\nget_env_registry_info name = " << it->second.env_str << ", name2 = " << get_env_registry()[env_var].env_str << std::endl;
    return it->second;
}*/

static string get_env_var_name(const ngraph::EnvVarEnum env_var)
{
    //string str = get_env_registry_info(env_var).env_str;
    string str = get_env_registry()[env_var].env_str;
    //std::cout << "Enum " << std::to_string(int(env_var)) << ", string name = " << str << "\n";
    return str;
}

static string get_env_var_default(const ngraph::EnvVarEnum env_var)
{
    //string def = get_env_registry_info(env_var).default_val;
    string def = get_env_registry()[env_var].default_val;
    //std::cout << "Enum " << std::to_string(int(env_var)) << ", default val = " << def << "\n";
    return def;
}

static string get_env_var_desc(const ngraph::EnvVarEnum env_var)
{
    return get_env_registry()[env_var].desc;
}

void ngraph::log_registry_envvar()
{
    NGRAPH_DEBUG << "List of all environment variables in registry:\n";
    for (uint32_t i = 0;  i < uint32_t(ngraph::EnvVarEnum::NGRAPH_ENV_VARS_COUNT); i++)
    {
        //NGRAPH_DEBUG << "\t" << get_env_var_name(it->first) << " = " << it->second << std::endl;
//        std::cout << "\tRL: Enum = " << std::to_string(int(it->first)) << ", name = " << get_env_registry()[ngraph::EnvVarEnum::NGRAPH_ENABLE_TRACING].env_str <<
//                                         ", default = " << get_env_registry()[ngraph::EnvVarEnum::NGRAPH_ENABLE_TRACING].default_val << std::endl;
        //std::cout << "\tRL: Enum = " << i << ", name = " << get_env_registry()[int(it->first)].env_str <<
        //                                 ", default = " << get_env_registry()[int(it->first)].default_val << std::endl;

        //std::cout << "\tRL: Enum = " << std::to_string(int(it->first)) << ", name = " << get_env_registry()[it->first].env_str << ", default = " << get_env_registry()[it->first].default_val << std::endl;
        //std::cout << "\tRL: Enum = " << std::to_string(int(it->first)) << ", name = " << get_env_var_name(it->first) << ", default = " << get_env_var_default(it->first) << std::endl;
        std::cout << "\tEnum = " << i << ", name = " << get_env_var_name(static_cast<ngraph::EnvVarEnum>(i) )<< ", default = " << get_env_var_default(static_cast<ngraph::EnvVarEnum>(i) ) << std::endl;
  //      it++;
    }
}

// Above this is registry related stuff
//--------------------------

// template <typename ET>
int ngraph::set_environment(const ngraph::EnvVarEnum env_var_enum, const char* value, const int overwrite)
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

    log_all_envvar();
#ifdef _WIN32
    return _putenv_s(env_var, value);
#elif defined(__linux) || defined(__APPLE__)
    return setenv(env_var, value, overwrite);
#endif
}

// template <typename ET>
int ngraph::unset_environment(const ngraph::EnvVarEnum env_var_enum)
{
    const char* env_var = get_env_var_name(env_var_enum).c_str();
    erase_env_from_cache(env_var_enum);
    log_all_envvar();
#ifdef _WIN32
    return _putenv_s(env_var, "");
#elif defined(__linux) || defined(__APPLE__)
    return unsetenv(env_var);
#endif
}
// set and unset programmatical apis
//----------------------

// --------- below this is caching apis
std::unordered_map<ngraph::EnvVarEnum, std::string>& get_env_var_cache()
{
    static std::unordered_map<ngraph::EnvVarEnum, string> s_env_var_cache;
    return s_env_var_cache;
}

void ngraph::log_all_envvar()
{
    NGRAPH_DEBUG << "List of all environment variables:\n";
    std::unordered_map<ngraph::EnvVarEnum, std::string>::iterator it = get_env_var_cache().begin();
    while (it != get_env_var_cache().end())
    {
        //NGRAPH_DEBUG << "\t" << get_env_var_name(it->first) << " = " << it->second << std::endl;
        std::cout << "\t" << get_env_var_name(it->first) << " = " << it->second << std::endl;
        it++;
    }
}

void ngraph::addenv_to_cache(const ngraph::EnvVarEnum env_var, const char* val)
{
    get_env_var_cache().emplace(env_var, val);
}

bool ngraph::env_cache_contains(const ngraph::EnvVarEnum env_var)
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

std::string ngraph::getenv_from_cache(const ngraph::EnvVarEnum env_var)
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

void ngraph::erase_env_from_cache(const ngraph::EnvVarEnum env_var)
{
    get_env_var_cache().erase(env_var);
}

std::string ngraph::getenv_string(const ngraph::EnvVarEnum env_var)
{
    if (env_cache_contains(env_var))
    {
        string env_string = getenv_from_cache(env_var);
        std::cout << "\t--- > getenv_string (from cache): " << get_env_var_name(env_var) << " = " << env_string << std::endl;
        return env_string;
    }
    else
    {
        const char* env_p = ::getenv(get_env_var_name(env_var).c_str());
        string env_string = env_p ? env_p : "";
        addenv_to_cache(env_var, env_string.c_str());
        std::cout << "\t--- > getenv_string (not cache): " << get_env_var_name(env_var) << " = " << env_string << std::endl;
        return env_string;
    }
}

int32_t ngraph::getenv_int(const ngraph::EnvVarEnum env_var/*, int32_t default_value*/)
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
            //NGRAPH_DEBUG << "Error reading (" << env_var << ") empty or undefined, "
            std::cout << "Error reading (" << get_env_var_name(env_var) << ") empty or undefined, "
                         << " defaulted to -1 here.";
            env_int = strtol(get_env_var_default(env_var).c_str(), &err, 0);
        }
        std::cout << "\t--- > getenv_int (from cache): " << get_env_var_name(env_var) << " = " << env_int << std::endl;
        return env_int;
    }
    else
    {
        //const char* env_p = ::getenv(env_var);
        const char* env_p = ::getenv(get_env_var_name(env_var).c_str());
        int32_t env_int = strtol(get_env_var_default(env_var).c_str(), &err, 0);//default_value;
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
                ss << "Environment variable \"" << get_env_var_name(env_var).c_str() << "\"=\"" << env_p
                   << "\" converted to different value \"" << env_int << "\" due to overflow."
                   << std::endl;
                throw runtime_error(ss.str());
            }
            // if syntax error is there - conversion will still happen
            // but warn user of syntax error
            if (*err)
            {
                std::stringstream ss;
                ss << "Environment variable \"" << get_env_var_name(env_var).c_str() << "\"=\"" << env_p
                   << "\" converted to different value \"" << env_int << "\" due to syntax error \""
                   << err << '\"' << std::endl;
                throw runtime_error(ss.str());
            }
        }
        else
        {
            NGRAPH_DEBUG << "Environment variable (" << env_var << ") empty or undefined, "
                         << " defaulted to -1 here.";
        }
        addenv_to_cache(env_var, std::to_string(env_int).c_str());
        std::cout << "\t--- > getenv_int (NOT cache): " << get_env_var_name(env_var) << " = " << env_int << std::endl;
        return env_int;
    }
}

bool ngraph::getenv_bool(const ngraph::EnvVarEnum env_var/*, bool default_value*/)
{
    string value = to_lower(getenv_string(env_var));
    std::cout << "get_bool() --- > getenv_bool (string value) " << get_env_var_name(env_var) << ", val = " << value << std::endl;
    static const set<string> off = {"0", "false", "off", "FALSE", "OFF", "no", "NO"};
    static const set<string> on = {"1", "true", "on", "TRUE", "ON", "yes", "YES"};
    bool rc = false;
    if (value == "")
    {
        rc = false; //default_value;
    }
    else if (off.find(value) != off.end())
    {
        rc = false;
    }
    else if (on.find(value) != on.end())
    {
        rc = true;
        std::cout << "\t\tsetting rc= true, get_bool() --- > in on section, value = " << value << " value ends\n";
    }
    else
    {
        stringstream ss;
        ss << "environment variable '" << get_env_var_name(env_var).c_str() << "' value '" << value
           << "' invalid. Must be boolean.";
        throw runtime_error(ss.str());
    }
    std::cout << "get_bool() --- > getenv_bool (bool value) " << get_env_var_name(env_var) << ", val = " << rc << "(line ends)" << std::endl;
    return rc;
}
