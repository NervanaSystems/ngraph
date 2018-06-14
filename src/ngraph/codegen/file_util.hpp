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
#include <string>
#include <vector>

namespace ngraph
{
    namespace codegen
    {
        class file_util;
    }
}

class ngraph::codegen::file_util
{
public:
    static std::string get_file_name(const std::string&);
    static std::string get_file_ext(const std::string&);
    static std::string path_join(const std::string& s1, const std::string& s2);
    static size_t get_file_size(const std::string& filename);
    static void remove_directory(const std::string& dir);
    static bool make_directory(const std::string& dir);
    static std::string make_temp_directory(const std::string& path = "");
    static std::string get_temp_directory();
    static void remove_file(const std::string& file);
    static std::vector<char> read_file_contents(const std::string& path);
    static std::string read_file_to_string(const std::string& path);
    static void iterate_files(const std::string& path,
                              std::function<void(const std::string& file, bool is_dir)> func,
                              bool recurse = false);
    static std::string tmp_filename(const std::string& extension = "");
    static void touch(const std::string& filename);
    static bool exists(const std::string& filename);
    static int try_get_lock(const std::string& filename);
    static void release_lock(int fd, const std::string& filename);
    static time_t get_timestamp(const std::string& filename);

private:
    static void iterate_files_worker(const std::string& path,
                                     std::function<void(const std::string& file, bool is_dir)> func,
                                     bool recurse = false);
};
