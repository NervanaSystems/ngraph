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

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "ngraph/function.hpp"
#include "ngraph/runtime/plaidml/plaidml_compiled_function.hpp"
#include "ngraph/runtime/plaidml/plaidml_compiler.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            class CompilationCache;
        }
    }
}

// A compilation cacher.
class ngraph::runtime::plaidml::CompilationCache final
{
public:
    // Looks up the supplied function in the compilation cache.  If the function is not in the
    // cache, returns an empty pointer.
    std::shared_ptr<CompiledFunction> try_lookup(std::shared_ptr<Function> func);

    // Looks up the supplied function in the compilation cache.  If the function is not in the
    // cache, compiles it using the specified compiler (which must not be nullptr), adds the
    // compiled function to the cache, and returns the compiled function.
    std::shared_ptr<CompiledFunction> compile(std::shared_ptr<Function> func, Compiler* compiler);

    // Drops the supplied function's compiled function from the compilation cache.
    void forget(std::shared_ptr<Function> func);

private:
    std::mutex m_mu;
    std::unordered_map<std::shared_ptr<Function>, std::shared_ptr<CompiledFunction>> m_cache;
};
