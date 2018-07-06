/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

std::string trim(const std::string& s);
std::vector<std::string> split(const std::string& src, char delimiter, bool do_trim = false);
bool is_version_number(const std::string& path);
std::string path_join(const std::string& s1, const std::string& s2);
std::string read_file_to_string(const std::string& path);
void iterate_files_worker(const std::string& path,
                          std::function<void(const std::string& file, bool is_dir)> func,
                          bool recurse);
void iterate_files(const std::string& path,
                   std::function<void(const std::string& file, bool is_dir)> func,
                   bool recurse);
std::string get_file_name(const std::string& s);
std::string get_file_ext(const std::string& s);
std::string to_hex(int value);
void dump(std::ostream& out, const void* vdata, size_t size);
time_t get_timestamp(const std::string& filename);

template <typename U, typename T>
bool contains(const U& container, const T& obj)
{
    bool rc = false;
    for (auto o : container)
    {
        if (o == obj)
        {
            rc = true;
            break;
        }
    }
    return rc;
}

template <typename T>
std::string join(const T& v, const std::string& sep = ", ")
{
    std::ostringstream ss;
    for (const auto& x : v)
    {
        if (&x != &*(v.begin()))
        {
            ss << sep;
        }
        ss << x;
    }
    return ss.str();
}
