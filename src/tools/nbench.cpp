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

// tool to benchmark any ngraph json model with given backend.
// compile and run with:
// g++ ./nbench.cpp -std=c++11 -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -lngraph -o nbench
// env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib env NGRAPH_INTERPRETER_EMIT_TIMING=1 ./nbench
// sample models are under ../../test/models

#include <fstream>
#include <ngraph/runtime/backend.hpp>
#include <ngraph/runtime/call_frame.hpp>
#include <ngraph/runtime/manager.hpp>
#include "../../test/util/benchmark.hpp"
#include "../../test/util/test_tools.hpp"
using namespace std;

int main(int argc, char** argv)
{
    string model = "model.json";
    string backend = "INTERPRETER";
    int iter = 10;
    bool failed = false;
    for (size_t i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "-f")
        {
            model = argv[++i];
        }
        else if (string(argv[i]) == "-b")
        {
            backend = argv[++i];
        }
        else if (string(argv[i]) == "-i")
        {
            try
            {
                iter = stoi(argv[++i]);
            }
            catch (...)
            {
                cout << "Invalid Argument\n";
                failed = true;
            }
        }
    }
    if (!static_cast<bool>(ifstream(model)))
    {
        cout << "File " << model << " not found\n";
        failed = true;
    }

    if (failed)
    {
        cout << R"###(
DESCRIPTION                                                         
    Benchmark ngraph json model with given backend.                 
                                                                    
SYNOPSIS                                                            
        nbench [-f <filename>] [-b <backend>] [-i <iterations>]
                                                                    
OPTIONS                                                             
        -f          model json file to use (default: model.json)    
        -b          Backend to use (default: INTERPRETER)           
        -i          Iterations (default: 10)                        
)###";
        return 1;
    }
    cout << "Benchmarking " << model << ", " << backend << " backend, " << iter << " iterations.\n";
    run_benchmark(model, backend, iter);
}
