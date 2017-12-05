// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "util.hpp"

using namespace std;

const string rewrite_header(const string& s, const string& path)
{
    stringstream ss(s);
    stringstream out;
    for (string line; ss; getline(ss, line))
    {
        // only interested in lines starging with '#include' so 8 chars minimum
        if (line.size() > 8 && line.substr(0, 8) == "#include")
        {
            auto line_offset = line.find_first_of("\"<");
            if (line_offset != string::npos)
            {
                string include = line.substr(line_offset);
                string contents = include.substr(1, include.size() - 2);
                if (include[1] == '.')
                {
                    if (include[2] == '/')
                    {
                        // include starts with './'
                        // rewrite "./blah.h" to "blah.h"
                        contents = contents.substr(2);
                    }
                    else
                    {
                        // include starts with '../'
                        // count number of '../' in string
                        size_t offset = 0;
                        size_t depth = 0;
                        while (contents.substr(offset, 3) == "../")
                        {
                            depth++;
                            offset += 3;
                        }
                        string trimmed = contents.substr(offset);
                        vector<string> parts = split(path, '/');
                        parts.pop_back();
                        size_t result_depth = parts.size() - depth;
                        string added_path;
                        for (size_t i = 0; i < result_depth; i++)
                        {
                            added_path += parts[i] + "/";
                        }
                        contents = added_path + trimmed;
                    }
                    if (include[0] == '<')
                    {
                        line = "#include <" + contents + ">";
                    }
                    else
                    {
                        line = "#include \"" + contents + "\"";
                    }
                }
            }
        }
        out << line << "\n";
    }
    return out.str();
}

class ResourceInfo
{
public:
    ResourceInfo(const string& source, const vector<string>& _subdirs, bool recursive = false)
        : search_path(source)
        , subdirs(_subdirs)
        , is_recursive(recursive)

    {
    }

    const string search_path;
    const vector<string> subdirs;
    const bool is_recursive;

    vector<string> files;
};

string find_path(const string& path)
{
    string rc;
    iterate_files(path,
                  [&](const string& file, bool is_dir) {
                      if (is_dir)
                      {
                          string dir_name = get_file_name(file);
                          if (is_version_number(dir_name))
                          {
                              rc = file;
                          }
                      }
                  },
                  true);
    return rc;
}

int main(int argc, char** argv)
{
    static vector<string> valid_ext = {".h", ".hpp", ".tcc", ""};
    string output_path;
    string base_name;

    for (size_t i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "--output")
        {
            output_path = argv[++i];
        }
        else if (string(argv[i]) == "--base_name")
        {
            base_name = argv[++i];
        }
    }

    string cpp0 = find_path("/usr/include/x86_64-linux-gnu/c++/");
    string cpp1 = find_path("/usr/include/c++/");

    cout << "Eigen path " << EIGEN_HEADERS_PATH << endl;
    cout << "ngraph path " << NGRAPH_HEADERS_PATH << endl;

    vector<ResourceInfo> include_paths;
    include_paths.push_back({CLANG_BUILTIN_HEADERS_PATH, {}, true});
    include_paths.push_back({"/usr/include/x86_64-linux-gnu", {"asm", "sys", "bits", "gnu"}});
    include_paths.push_back({"/usr/include", {"linux", "asm-generic"}});
    include_paths.push_back({cpp0, {"bits"}});
    include_paths.push_back({cpp1, {"bits", "ext", "debug", "backward"}});
    include_paths.push_back({EIGEN_HEADERS_PATH, {}, true});
    include_paths.push_back({NGRAPH_HEADERS_PATH, {}, true});

    if (output_path.empty())
    {
        cout << "must specify output path with --output option" << endl;
        return -1;
    }

    auto output_timestamp = get_timestamp(output_path);

    for (ResourceInfo& path : include_paths)
    {
        // cout << "path " << path.source_path << " -> " << path.target_path << endl;
        vector<string> path_list;
        path_list.push_back(path.search_path);
        for (const string& p : path.subdirs)
        {
            path_list.push_back(path_join(path.search_path, p));
        }
        for (const string& p : path_list)
        {
            iterate_files(p,
                          [&](const string& file, bool is_dir) {
                              if (!is_dir)
                              {
                                  string ext = get_file_ext(file);
                                  if (contains(valid_ext, ext))
                                  {
                                      //   cout << "add " << path.search_path << ", " << file << endl;
                                      path.files.push_back(file);
                                  }
                              }
                          },
                          path.is_recursive);
        }
    }

    // test for changes to any headers
    bool update_needed = true;
    for (ResourceInfo& path : include_paths)
    {
        for (const string& header_file : path.files)
        {
            auto file_timestamp = get_timestamp(header_file);
            if (file_timestamp > output_timestamp)
            {
                update_needed = true;
                break;
            }
        }
    }

    if (update_needed)
    {
        ofstream out(output_path);
        out << "#pragma clang diagnostic ignored \"-Weverything\"\n";
        out << "#include <vector>\n";
        out << "namespace ngraph\n";
        out << "{\n";
        out << "    uint8_t header_resources[] =\n";
        out << "    {\n";
        vector<pair<size_t, size_t>> offset_size_list;
        size_t offset = 0;
        for (const ResourceInfo& path : include_paths)
        {
            for (const string& header_file : path.files)
            {
                // string search_path =
                //     path.target_path.substr(0, path.target_path.find_first_of("/", 1));
                // string source_full_path = path_join(header_file.first, header_file.second);
                string header_data = read_file_to_string(header_file);
                string base_path = header_file.substr(path.search_path.size() + 1);
                // header_data = rewrite_header(header_data, base_path);
                // string target_path = path_join(path.target_path, header_file.second);

                // data layout is triplet of strings containing:
                // 1) search path
                // 2) header path within search path
                // 3) header data
                // all strings are null terminated and the length includes the null
                // The + 1 below is to account for the null terminator

                dump(out, path.search_path.c_str(), path.search_path.size() + 1);
                offset_size_list.push_back({offset, path.search_path.size() + 1});
                offset += path.search_path.size() + 1;

                dump(out, header_file.c_str(), header_file.size() + 1);
                offset_size_list.push_back({offset, header_file.size() + 1});
                offset += header_file.size() + 1;

                dump(out, header_data.c_str(), header_data.size() + 1);
                offset_size_list.push_back({offset, header_data.size() + 1});
                offset += header_data.size() + 1;
            }
        }
        out << "    };\n";
        out << "    struct HeaderInfo\n";
        out << "    {\n";
        out << "        const char* search_path;\n";
        out << "        const char* header_path;\n";
        out << "        const char* header_data;\n";
        out << "    };\n";
        out << "    std::vector<HeaderInfo> header_info\n";
        out << "    {\n";
        for (size_t i = 0; i < offset_size_list.size();)
        {
            out << "        {(char*)(&header_resources[" << offset_size_list[i++].first;
            out << "]), (char*)(&header_resources[" << offset_size_list[i++].first;
            out << "]), (char*)(&header_resources[" << offset_size_list[i++].first << "])},\n";
        }
        out << "    };\n";
        out << "}\n";
    }
}
