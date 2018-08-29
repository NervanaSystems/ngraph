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

#include <llvm/ExecutionEngine/ExecutionEngine.h>

#include "ngraph/codegen/execution_engine.hpp"

using namespace ngraph;

codegen::ExecutionEngine::ExecutionEngine()
    : m_execution_engine{nullptr}
{
}

codegen::ExecutionEngine::~ExecutionEngine()
{
    if (m_execution_engine)
    {
        m_execution_engine->runStaticConstructorsDestructors(true);
    }
}

bool codegen::ExecutionEngine::add_module(std::unique_ptr<ngraph::codegen::Module>& module)
{
    if (module)
    {
        if (!m_execution_engine)
        {
            m_execution_engine.reset(llvm::EngineBuilder(module->take_module())
                                         .setEngineKind(llvm::EngineKind::JIT)
                                         .setOptLevel(llvm::CodeGenOpt::Aggressive)
                                         .setMCPU(llvm::sys::getHostCPUName())
                                         //  .setCodeModel(llvm::CodeModel::Medium)
                                         .setErrorStr(&m_jit_error)
                                         .create());

            if (!m_execution_engine)
            {
                return false;
            }
        }
    }
    else
    {
        return false;
    }

    return true;
}

void codegen::ExecutionEngine::finalize()
{
    if (m_execution_engine)
    {
        m_execution_engine->finalizeObject();
        m_execution_engine->runStaticConstructorsDestructors(false);
    }
    else
    {
        throw std::runtime_error(
            "Error in finalize: " +
            (m_jit_error.empty() ? "Could not create an execution engine" : m_jit_error));
    }
}

void* codegen::ExecutionEngine::get_pointer_to_named_function(const std::string& func_name)
{
// For whatever reason, macOS seems to expect that we prefix this with an underscore.
#ifdef __APPLE__
    std::string fname = "_" + func_name;
#else
    const std::string& fname = func_name;
#endif

    // set AbortOnFailure flag to false so call fails by returning nullptr
    return m_execution_engine->getPointerToNamedFunction(fname, false);
}
