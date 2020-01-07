//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "header_rewrite.hpp"
#include "uncomment.hpp"
#include "util.hpp"

using namespace std;

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
    time_t main_timestamp = get_timestamp(argv[0]);
    static vector<string> valid_ext = {".h", ".hpp", ".tcc", ""};
    static vector<string> invalid_file = {"README"};
    string output_path;
    string base_name;

    for (int64_t i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "--output")
        {
            output_path = argv[++i];
        }
        else if (arg == "--base_name")
        {
            base_name = argv[++i];
        }
    }

    vector<ResourceInfo> include_paths;

    include_paths.push_back({CLANG_BUILTIN_HEADERS_PATH, {}, true});

#ifdef EIGEN_HEADERS_PATH
    include_paths.push_back({EIGEN_HEADERS_PATH, {"Eigen"}, true});
#endif
#ifdef MKLDNN_HEADERS_PATH
    include_paths.push_back({MKLDNN_HEADERS_PATH, {}, true});
#endif
#ifdef TBB_HEADERS_PATH
    include_paths.push_back({TBB_HEADERS_PATH, {}, true});
#endif
    include_paths.push_back({NGRAPH_HEADERS_PATH, {"ngraph"}, true});

    if (output_path.empty())
    {
        cout << "must specify output path with --output option" << endl;
        return -1;
    }

    time_t output_timestamp = get_timestamp(output_path);

    for (ResourceInfo& path : include_paths)
    {
        vector<string> path_list;
        if (path.subdirs.empty())
        {
            path_list.push_back(path.search_path);
        }
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
                                  if (!contains(invalid_file, file))
                                  {
                                      string ext = get_file_ext(file);
                                      if (contains(valid_ext, ext))
                                      {
                                          // std::cout << "add " << file << std::endl;
                                          path.files.push_back(file);
                                      }
                                  }
                              }
                          },
                          path.is_recursive);
        }
    }

    // test for changes to any headers
    bool update_needed = main_timestamp > output_timestamp;
    if (!update_needed)
    {
        for (ResourceInfo& path : include_paths)
        {
            for (const string& header_file : path.files)
            {
                time_t file_timestamp = get_timestamp(header_file);
                if (file_timestamp > output_timestamp)
                {
                    update_needed = true;
                    break;
                }
            }
        }
    }

    if (update_needed)
    {
        size_t total_size = 0;
        size_t total_count = 0;
        const string delim = "pReFiX";
        ofstream out(output_path);
        out << "#pragma clang diagnostic ignored \"-Weverything\"\n";
        out << "#include <vector>\n";
        out << "namespace ngraph\n";
        out << "{\n";
        out << "    const std::vector<std::string> builtin_search_paths =\n";
        out << "    {\n";
        for (const ResourceInfo& path : include_paths)
        {
            out << "        \"" << path.search_path << "\",\n";
        }
        out << "    };\n";
#ifdef _WIN32
        out << "    const std::vector<std::pair<std::string, std::vector<std::string>>> "
               "builtin_headers =\n";
#else
        out << "    const std::vector<std::pair<std::string, std::string>> builtin_headers =\n";
#endif
        out << "    {\n";
        for (const ResourceInfo& path : include_paths)
        {
            for (const string& header_path : path.files)
            {
                out << "        {";
                out << "\"" << header_path << "\",\n";
                string relative_path = header_path.substr(path.search_path.size() + 1);
                std::ifstream file(header_path);
                if (file.is_open())
                {
                    std::string line;
#ifdef _WIN32
                    const int max_partial_size = 65500;
                    out << "{\n";
                    bool first_line = true;
                    int partial_size = 0;
                    out << "R\"" << delim << "(";
                    while (getline(file, line))
                    {
                        line = rewrite_header(line, relative_path);
                        // line = uncomment(line);
                        total_size += line.size();
                        partial_size += line.size();
                        if (partial_size > max_partial_size)
                        {
                            out << ")" << delim << "\",\n";
                            partial_size = line.size();
                            out << "R\"" << delim << "(";
                        }
                        out << line;
                    }
                    out << ")" << delim << "\"";
                    out << "}";
#else
                    out << "R\"" << delim << "(";
                    while (getline(file, line))
                    {
                        line = rewrite_header(line, relative_path);
                        // line = uncomment(line);
                        total_size += line.size();
                        out << line;
                    }
                    out << ")" << delim << "\"";
#endif
                    file.close();
                }
                out << "},\n";
                total_count++;
            }
        }
        out << "    };\n";
        out << "}\n";
        cout.imbue(locale(""));
        cout << "Total size " << total_size << " in " << total_count << " files\n";
    }
    return 0;
}
