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

#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"

namespace ngraph
{
    namespace op
    {
        class FunctionCall : public Builtin
        {
        public:
            ///
            /// @param function The function to be called
            /// @param args The function arguments
            ///
            FunctionCall(const std::shared_ptr<Function>&          function,
                         const std::vector<std::shared_ptr<Node>>& args)
                : Builtin(args)
                , m_function(function)
            {
            }

            virtual std::string description() const override { return "FunctionCall"; }
            virtual void        propagate_types() override;

            std::shared_ptr<Function> get_function() const { return m_function; }

        protected:
            std::shared_ptr<Function> m_function;
        };
    }
}
