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

#include <iostream>
#include <sstream>

#include "uncomment.hpp"

using namespace std;

// start 23,749,645 in 1,912 files

void skip_comment(istream& s)
{
}

string uncomment(const string& s)
{
    stringstream out;
    for (size_t i = 0; i < s.size(); i++)
    {
        char c = s[i];
        if (i < s.size() - 1 && c == '/' && s[i + 1] == '/')
        {
            while (i < s.size() && c != '\n')
            {
                c = s[++i];
            }
            out << "\n";
        }
        else
        {
            out << c;
        }
    }
    return out.str();
}
