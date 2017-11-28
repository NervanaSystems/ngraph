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

class ResourceInfo
{
public:
    ResourceInfo(const string& source, const string& target, bool recursive = false)
        : source_path(source)
        , target_path(target)
        , is_recursive(recursive)

    {
    }

    const string source_path;
    const string target_path;
    const bool is_recursive;

    vector<pair<string, string>> files;
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
    include_paths.push_back({CLANG_BUILTIN_HEADERS_PATH, "/$builtin0", true});
    include_paths.push_back({"/usr/include/x86_64-linux-gnu", "/$builtin1"});
    include_paths.push_back({"/usr/include/x86_64-linux-gnu/asm", "/$builtin1/asm"});
    include_paths.push_back({"/usr/include/x86_64-linux-gnu/sys", "/$builtin1/sys"});
    include_paths.push_back({"/usr/include/x86_64-linux-gnu/bits", "/$builtin1/bits"});
    include_paths.push_back({"/usr/include/x86_64-linux-gnu/gnu", "/$builtin1/gnu"});
    include_paths.push_back({"/usr/include", "/$builtin2"});
    include_paths.push_back({"/usr/include/linux", "/$builtin2/linux"});
    include_paths.push_back({"/usr/include/asm-generic", "/$builtin2/asm-generic"});
    include_paths.push_back({cpp0, "/$builtin3"});
    include_paths.push_back({cpp0 + "/bits", "/$builtin3/bits"});
    include_paths.push_back({cpp1, "/$builtin4"});
    include_paths.push_back({cpp1 + "/bits", "/$builtin4/bits"});
    include_paths.push_back({cpp1 + "/ext", "/$builtin4/ext"});
    include_paths.push_back({cpp1 + "/debug", "/$builtin4/debug"});
    include_paths.push_back({cpp1 + "/backward", "/$builtin4/backward"});
    include_paths.push_back({EIGEN_HEADERS_PATH, "/#builtin5", true});
    include_paths.push_back({NGRAPH_HEADERS_PATH, "/#builtin6", true});

    if (output_path.empty())
    {
        cout << "must specify output path with --output option" << endl;
        return -1;
    }

    auto output_timestamp = get_timestamp(output_path);

    for (ResourceInfo& path : include_paths)
    {
        // cout << "path " << path.source_path << " -> " << path.target_path << endl;
        iterate_files(path.source_path,
                      [&](const string& file, bool is_dir) {
                          if (!is_dir)
                          {
                              string trimmed = file.substr(path.source_path.size() + 1);
                              string ext = get_file_ext(trimmed);
                              if (contains(valid_ext, ext))
                              {
                                  //   cout << "add " << path.source_path << ", " << trimmed << endl;
                                  path.files.push_back({path.source_path, trimmed});
                              }
                          }
                      },
                      path.is_recursive);
    }

    // test for changes to any headers
    bool update_needed = false;
    for (ResourceInfo& path : include_paths)
    {
        for (const pair<string, string>& header_file : path.files)
        {
            string source_full_path = path_join(header_file.first, header_file.second);
            auto file_timestamp = get_timestamp(source_full_path);
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
            for (const pair<string, string>& header_file : path.files)
            {
                string source_full_path = path_join(header_file.first, header_file.second);
                string header_data = read_file_to_string(source_full_path);
                string target_path = path_join(path.target_path, header_file.second);
                string search_path =
                    path.target_path.substr(0, path.target_path.find_first_of("/", 1));

                // data layout is triplet of strings containing:
                // 1) search path
                // 2) header path within search path
                // 3) header data
                // all strings are null terminated and the length includes the null
                // The + 1 below is to account for the null terminator

                dump(out, search_path.c_str(), search_path.size() + 1);
                offset_size_list.push_back({offset, search_path.size() + 1});
                offset += search_path.size() + 1;

                dump(out, target_path.c_str(), target_path.size() + 1);
                offset_size_list.push_back({offset, target_path.size() + 1});
                offset += target_path.size() + 1;

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
