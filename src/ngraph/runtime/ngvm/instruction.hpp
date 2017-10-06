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

namespace ngraph
{
    namespace runtime
    {
        namespace ngvm
        {
            class CallFrame;

            /// @brief An interpreter for an Op
            ///
            /// The call_frame has a vector of instructions and calls execute on each instruction, passing it the call_frame.
            /// Instructions get argument, result, and intermediate tensor views from the call frame. Instructions may also
            /// set a flag in the call_frame to end execution, or adjust execution by modifying the position in the instruction vector.
            class Instruction
            {
            public:
                virtual ~Instruction() {}
                virtual void execute(CallFrame& call_frame) const = 0;
            };
        }
    }
}
