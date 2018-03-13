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

#include <fstream>
#include <ngraph/codegen/compiler.hpp>
#include <ngraph/codegen/execution_engine.hpp>
#include <ngraph/file_util.hpp>
#include <ngraph/util.hpp>
#include <ngraph/util.hpp>

using namespace std;
using namespace ngraph;

void help()
{
    cout << R"###(
DESCRIPTION
    Benchmark compile process identical to ngraph JIT.

SYNOPSIS
        compile_benchmark <filename>
)###" << endl;
}

int main(int argc, char** argv)
{
    string source_path;
    for (size_t i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            help();
        }
        else
        {
            source_path = arg;
        }
    }

    if (!file_util::exists(source_path))
    {
        cout << "file '" << source_path << "' not found\n";
        help();
        return 1;
    }
    else
    {
        stopwatch timer;

        const string source_string = file_util::read_file_to_string(source_path);
        codegen::Compiler compiler;
        codegen::ExecutionEngine engine;

        timer.start();
        auto module = compiler.compile(source_string);
        timer.stop();
        cout << "compile took " << timer.get_milliseconds() << "ms\n";

        timer.start();
        engine.add_module(module);
        engine.finalize();
        timer.stop();
        cout << "execution engine took " << timer.get_milliseconds() << "ms\n";
    }

    return 0;
}
