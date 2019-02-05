//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "backend.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief Representation of onnxGraph
        class Graph
        {
        public:
            Graph(const Graph&) = delete;
            Graph& operator=(const Graph&) = delete;

            Graph() = delete;

            Graph(Graph&&) noexcept = default;
            Graph& operator=(Graph&&) noexcept = delete;

            explicit Graph(const Backend& backend)
                : m_backend{&backend}
            {
            }

        private:
            const Backend* m_backend;
        };

    } // namespace onnxifi

} // namespace ngraph
