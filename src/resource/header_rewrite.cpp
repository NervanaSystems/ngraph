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

#include <sstream>
#include <vector>

#include "header_rewrite.hpp"
#include "util.hpp"

using namespace std;

// This function rewrites all of the
// #include "../../blah"
// into something with a dotless relative path. It seems that clang can't handle the .. stuff
// when the header files are stored in its in-memory filesystem.
// Eigen has a lot of .. in their header files.
const string rewrite_header(const string& s, const string& path)
{
    stringstream ss(s);
    stringstream out;
    for (string line; ss; getline(ss, line))
    {
        string original_line = line;
        // only interested in lines starging with '#include' so 8 chars minimum
        if (line.size() > 8)
        {
            // skip whitespace
            size_t pos = line.find_first_not_of(" \t");
            if (pos != string::npos && line[pos] == '#' && pos < line.size() - 7)
            {
                string directive = line;
                pos = directive.find_first_not_of(" \t", pos + 1);
                if (pos != string::npos)
                {
                    directive = directive.substr(pos);
                }
                pos = directive.find_first_of(" \t", pos + 1);
                directive = directive.substr(0, pos);
                if (directive == "include")
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
                            // cout << "line '" << original_line << "'\n";
                            // cout << "rewrite to '" << line << "'\n\n";
                        }
                    }
                }
            }
        }
        out << line << "\n";
    }
    return out.str();
}
