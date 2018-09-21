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

#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "util.hpp"

using namespace std;

string trim(const string& s)
{
    string rc = s;
    // trim trailing spaces
    size_t pos = rc.find_last_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(0, pos + 1);
    }

    // trim leading spaces
    pos = rc.find_first_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(pos);
    }
    return rc;
}

vector<string> split(const string& src, char delimiter, bool do_trim)
{
    size_t pos;
    string token;
    size_t start = 0;
    vector<string> rc;
    while ((pos = src.find(delimiter, start)) != std::string::npos)
    {
        token = src.substr(start, pos - start);
        start = pos + 1;
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    if (start <= src.size())
    {
        token = src.substr(start);
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    return rc;
}

bool is_version_number(const string& path)
{
    bool rc = true;
    vector<string> tokens = split(path, '.');
    for (string s : tokens)
    {
        for (char c : s)
        {
            if (!isdigit(c))
            {
                rc = false;
            }
        }
    }
    return rc;
}

string path_join(const string& s1, const string& s2)
{
    string rc;
    if (s2.size() > 0)
    {
        if (s2[0] == '/')
        {
            rc = s2;
        }
        else if (s1.size() > 0)
        {
            rc = s1;
            if (rc[rc.size() - 1] != '/')
            {
                rc += "/";
            }
            rc += s2;
        }
        else
        {
            rc = s2;
        }
    }
    else
    {
        rc = s1;
    }
    return rc;
}

std::string read_file_to_string(const std::string& path)
{
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void iterate_files_worker(const string& path,
                          std::function<void(const string& file, bool is_dir)> func,
                          bool recurse)
{
    DIR* dir;
    struct dirent* ent;

    // If we cannot open the directory, we silently ignore it.
    if ((dir = opendir(path.c_str())) != nullptr)
    {
        while ((ent = readdir(dir)) != nullptr)
        {
            string name = ent->d_name;
            switch (ent->d_type)
            {
            case DT_DIR:
                if (name != "." && name != "..")
                {
                    string dir_path = path_join(path, name);
                    if (recurse)
                    {
                        iterate_files(dir_path, func, recurse);
                    }
                    func(dir_path, true);
                }
                break;
            case DT_LNK: break;
            case DT_REG:
            {
                string file_name = path_join(path, name);
                func(file_name, false);
                break;
            }
            default: break;
            }
        }
        closedir(dir);
    }
}

void iterate_files(const string& path,
                   std::function<void(const string& file, bool is_dir)> func,
                   bool recurse)
{
    vector<string> files;
    vector<string> dirs;
    iterate_files_worker(path,
                         [&files, &dirs](const string& file, bool is_dir) {
                             if (is_dir)
                                 dirs.push_back(file);
                             else
                                 files.push_back(file);
                         },
                         recurse);

    for (auto f : files)
    {
        func(f, false);
    }
    for (auto f : dirs)
    {
        func(f, true);
    }
}

std::string get_file_name(const std::string& s)
{
    string rc = s;
    auto pos = s.find_last_of('/');
    if (pos != string::npos)
    {
        rc = s.substr(pos + 1);
    }
    return rc;
}

std::string get_file_ext(const std::string& s)
{
    string rc = get_file_name(s);
    auto pos = rc.find_last_of('.');
    if (pos != string::npos)
    {
        rc = rc.substr(pos);
    }
    else
    {
        rc = "";
    }
    return rc;
}

string to_hex(int value)
{
    stringstream ss;
    ss << "0x" << std::hex << std::setw(2) << std::setfill('0') << value;
    return ss.str();
}

void dump(ostream& out, const void* vdata, size_t size)
{
    const uint8_t* data = reinterpret_cast<const uint8_t*>(vdata);
    size_t index = 0;
    while (index < size)
    {
        out << "        ";
        for (size_t i = 0; i < 16 && index < size; i++)
        {
            if (i != 0)
            {
                out << ", ";
            }
            out << to_hex(data[index++]);
        }
        out << ",\n";
    }
}

time_t get_timestamp(const std::string& filename)
{
    time_t rc = 0;
    struct stat st;
    if (stat(filename.c_str(), &st) == 0)
    {
        rc = st.st_mtime;
    }
    return rc;
}
