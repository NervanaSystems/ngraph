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

#include <list>
#include <memory>
#include <typeinfo>
#include <vector>

#include "ngraph/pass/manager_state.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class Manager;
        class ManagerState;
    }
}

class ngraph::pass::Manager
{
public:
    Manager();
    ~Manager();

    void initialize_default_passes();

    template <typename T, class... Args>
    void register_pass(Args&&... args)
    {
        static_assert(std::is_base_of<pass::PassBase, T>::value, "pass not derived from pass base");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_base = std::static_pointer_cast<PassBase>(pass);
        m_pass_list.push_back(pass_base);
        if (m_visualize || m_serialize)
        {
            m_pass_names.push_back(typeid(T).name());
        }
    }

    void run_passes(std::shared_ptr<Function>, bool transitive = true);

    ManagerState& get_state();
    void set_pass_visualization(bool new_state) { m_visualize = new_state; }
    void set_pass_serialization(bool new_state) { m_serialize = new_state; }
private:
    std::vector<std::string> m_pass_names;
    std::vector<std::shared_ptr<PassBase>> m_pass_list;
    ManagerState m_state;
    bool m_visualize = false;
    bool m_serialize = false;
};
