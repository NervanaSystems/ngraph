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

#pragma once

#include <list>
#include <memory>
#include <vector>

#include "ngraph/pass/call_pass.hpp"
#include "ngraph/pass/manager_state.hpp"

namespace ngraph
{
    namespace pass
    {
        class Manager;
        class ManagerState;
    }

    class Node;
    class Function;
}

class ngraph::pass::Manager
{
public:
    Manager();
    ~Manager();

    void initialize_default_passes();

    template <typename T, class... Args>
    void register_pass(Args... args)
    {
        static_assert(std::is_base_of<pass::Base, T>::value, "pass not derived from pass base");
        if (std::is_base_of<CallBase, T>::value)
        {
            register_pass_ptr(std::make_shared<T>(args...));
        }
        else if (std::is_base_of<FunctionPass, T>::value)
        {
            register_pass_ptr(std::make_shared<T>(args...));
        }
    }

    void run_passes(Function*);
    void run_passes(std::shared_ptr<Function>);

    ManagerState& get_state();

private:
    void register_pass_ptr(std::shared_ptr<CallBase>);
    void register_pass_ptr(std::shared_ptr<FunctionPass>);

    std::vector<std::shared_ptr<CallBase>> m_call_passes;
    std::vector<std::shared_ptr<FunctionPass>> m_function_passes;
    ManagerState m_state;
};
