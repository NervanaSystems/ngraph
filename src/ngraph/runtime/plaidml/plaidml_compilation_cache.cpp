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

#include "ngraph/runtime/plaidml/plaidml_compilation_cache.hpp"

std::shared_ptr<ngraph::runtime::plaidml::CompiledFunction>
    ngraph::runtime::plaidml::CompilationCache::try_lookup(std::shared_ptr<Function> func)
{
    std::lock_guard<std::mutex> lock{m_mu};
    auto it = m_cache.find(func);
    if (it != m_cache.end())
    {
        return it->second;
    }
    return std::shared_ptr<CompiledFunction>{};
}

std::shared_ptr<ngraph::runtime::plaidml::CompiledFunction>
    ngraph::runtime::plaidml::CompilationCache::compile(std::shared_ptr<Function> func,
                                                        Compiler* compiler)
{
    std::lock_guard<std::mutex> lock{m_mu};
    auto it_inserted = m_cache.insert(std::make_pair(func, std::shared_ptr<CompiledFunction>{}));
    if (it_inserted.second)
    {
        try
        {
            it_inserted.first->second = compiler->compile(func);
        }
        catch (...)
        {
            m_cache.erase(it_inserted.first);
            throw;
        }
    }
    return it_inserted.first->second;
}

void ngraph::runtime::plaidml::CompilationCache::forget(std::shared_ptr<Function> func)
{
    std::lock_guard<std::mutex> lock{m_mu};
    m_cache.erase(func);
}
