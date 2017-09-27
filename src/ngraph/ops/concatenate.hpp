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

namespace ngraph
{
    namespace op
    {
        class Concat : public Builtin
        {
        public:
            /// Concatenates one or more tensors.
            ///
            /// All tensors must have the same rank, and the sizes of the axes must match
            /// everywhere except at the concatenation axis. The size of the concatenation
            /// axis on the output is the sum of its size on all inputs; the size of other
            /// axes is unchanged from the input tensors.
            ///
            /// Example: n0 has shape {2,4,2}, and n1 has shape {2,5,2}. Then the output of
            ///          Concat(Nodes{n0,n1},1) will have shape {2,9,2}.
            Concat(const Nodes& args,size_t concatenation_axis)
                : Builtin(args)
                , m_concatenation_axis(concatenation_axis)
            {
            }

            virtual std::string description() const override { return "Concatenate"; }
            virtual void        propagate_types() override;

            size_t get_concatenation_axis() const { return m_concatenation_axis; }

        protected:
            const size_t m_concatenation_axis;
        };
    }
}
