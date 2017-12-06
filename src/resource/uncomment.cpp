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

#include <sstream>

#include "uncomment.hpp"

using namespace std;

// start 23,749,645 in 1,912 files

void skip_comment(istream& s)
{
}

string uncomment(const string& s)
{
    stringstream ss(s);
    stringstream out;
    while (ss)
    {
        char c;
        ss >> c;
        out << c;
    }
    // for (string line; ss; getline(ss, line))
    // {
    //     for (size_t i = 0; i < line.size(); i++)
    //     {
    //         if (i < line.size() - 2 && line[i] == '/' && line[i + 1] == '/')
    //         {
    //             // start of a line comment
    //             out << '\n';
    //             break;
    //         }
    //         out << line[i];
    //     }
    // }
    return out.str();
}
