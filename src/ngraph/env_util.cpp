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
#include "ngraph/env_util.hpp"
#include "ngraph/util.hpp"

using namespace std;

static std::unordered_map<string, string> s_env_map;
static const size_t MAX_NUM_ENV_VARS = 100;

void ngraph::log_all_envvar()
{
    NGRAPH_DEBUG << "List of all environment variables:\n";
    std::unordered_map<std::string, std::string>::iterator it = s_env_map.begin();
    while (it != s_env_map.end())
    {
        NGRAPH_DEBUG << "\t" << it->first << " = " << it->second << std::endl;
        it++;
    }
}

void ngraph::addenv_to_map(std::string env_var, std::string val)
{
    if (s_env_map.size() <= MAX_NUM_ENV_VARS)
    {
        s_env_map.emplace(env_var, val);
    }
    else
    {
        NGRAPH_WARN << "Number of environment variables used is > max = " << MAX_NUM_ENV_VARS
                    << std::endl;
        log_all_envvar();
    }
}

bool ngraph::getenv_from_map(const char* env_var, std::string val)
{
    if (s_env_map.find(env_var) != s_env_map.end())
    {
        val = s_env_map.at(env_var);
        return true;
    }
    else
    {
        return false;
    }
}

std::string ngraph::getenv_string(const char* env_var)
{
    string env_string = "";
    if (!getenv_from_map(env_var, env_string))
    {
        const char* env_p = ::getenv(env_var);
        env_string = env_p ? env_p : "";
        addenv_to_map(env_var, env_string);
    }
    return env_string;
}

int32_t ngraph::getenv_int(const char* env_var, int32_t default_value)
{
    char* env_string;
    if (!getenv_from_map(env_var, env_string))
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
        // insert into map
        addenv_to_map(env_var, std::to_string(env));
        return env;
    }
    else
    {
        errno = 0;
        char* err;
        int32_t env_int = strtol(env_string, &err, 0);
        if (errno == 0 || *err)
        {
            // Extensive error checking was done when reading getenv, keeping it minimal here, ok?
            NGRAPH_DEBUG << "Error reading (" << env_var << ") empty or undefined, "
                         << " defaulted to -1 here.";
            return default_value;
        }
        return env_int;
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
