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
#include <iostream>
#include <regex>
#include <string>

using namespace std;

string generate_filename(string name, string version)
{
    stringstream ss;
    for (size_t i = 0; i < name.size(); ++i)
    {
        char c = name[i];
        if (isupper(c))
        {
            c = tolower(c);
            if (i > 0 && i < name.size() - 1 && !isupper(name[i + 1]))
            {
                ss << "_";
            }
        }
        ss << c;
    }
    ss << "_v" << version;
    return ss.str();
}

void generate_file(string name, string version)
{
    string filename = generate_filename(name, version);

    string source =
        R"(//*****************************************************************************
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

#include "contrib/mlir/core/pass/ng_dialect_builder.hpp"
#include "ngraph/ops.hpp"

template <>
mlir::Operation* ngraph::pass::NgDialectConversionPass::createOp<ngraph::op::v@VER::@OP>(
    NgDialectConversionPass& NgDialectObj, const ngraph::Node* ngNode)
{
    auto node = dynamic_cast<const ngraph::op::v@VER::@OP*>(ngNode);
    NGRAPH_CHECK(ngNode, node != nullptr, "ngNode ", ngNode->description(), " is not a v@VER::@OP");
    throw unsupported_op("Unsupported op 'v@VER::@OP'");
}
)";

    regex r1("@OP");
    regex r2("@VER");
    source = regex_replace(source, r1, name);
    source = regex_replace(source, r2, version);

    filename =
        "/nfs/pdx/home/rhkimbal/dev/ngraph/src/contrib/mlir/core/pass/convert/" + filename + ".cpp";
    cout << filename << endl;

    ofstream f(filename);
    f << source;
}

int main(int argc, const char** argv)
{
#define NGRAPH_OP(OP, VER) generate_file(#OP, #VER);
#include "ngraph/op_version_tbl.hpp"
    return 0;
}
