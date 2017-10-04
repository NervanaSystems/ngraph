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

#include <memory>
#include <vector>

namespace ngraph
{
    namespace pass
    {
        class Base;
        class FunctionPass;
        class Manager;
        class ManagerState;
    }
    class Function;
}

class ngraph::pass::Base
{
    friend class Manager;

public:
    virtual ~Base() {}
protected:
    ManagerState& get_state();
    void set_state(ManagerState&);

private:
    ManagerState* m_state;
};

class ngraph::pass::FunctionPass : public Base
{
public:
    virtual ~FunctionPass() {}
    virtual bool run_on_function(ngraph::Function*) = 0;

    // derived class throws exception if its dependencies have not been met
    virtual void check_dependencies(const std::vector<std::shared_ptr<FunctionPass>>&) const {}
};
