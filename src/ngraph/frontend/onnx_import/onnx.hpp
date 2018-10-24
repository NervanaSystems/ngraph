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

#include <iostream>
#include <string>
#include <unordered_map>

#include "ngraph/function.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"

#include "core/operator_set.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class Weight
        {
        public:
            enum class Type
            {
                f16,
                f32,
                f64,
                i8,
                i16,
                i32,
                i64,
                u8,
                u16,
                u32,
                u64
            };

            Weight() = delete;
            Weight(Type type, std::size_t dimensions, const std::size_t* shape, const void* data);

            Weight(Weight&&) noexcept = default;
            Weight& operator=(Weight&&) noexcept = default;

            Weight(const Weight&);
            Weight& operator=(const Weight&);

            ~Weight() = default;

            const element::Type& type() const;
            const void* data() const;
            const Shape& shape() const;

        private:
            struct Impl;
            std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl;
        };

        using Weights = std::unordered_map<std::string, Weight>;

        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn);

        // Convert on ONNX model to a vector of nGraph Functions (input stream)
        std::vector<std::shared_ptr<Function>> load_onnx_model(std::istream&,
                                                               const Weights& weights = {});

        // Convert an ONNX model to a vector of nGraph Functions
        std::vector<std::shared_ptr<Function>> load_onnx_model(const std::string&,
                                                               const Weights& weights = {});

        // Convert the first output of an ONNX model to an nGraph Function (input stream)
        std::shared_ptr<Function> import_onnx_function(std::istream&, const Weights& weights = {});

        // Convert the first output of an ONNX model to an nGraph Function
        std::shared_ptr<Function> import_onnx_function(const std::string&,
                                                       const Weights& weights = {});

    } // namespace onnx_import

} // namespace ngraph
