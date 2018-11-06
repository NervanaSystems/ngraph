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

#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "ngraph/runtime/gpu/gpu_host_parameters.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            template <typename T>
            struct has_const_iterator;
            template <typename T>
            struct is_container;

            class GPUKernelArgs
            {
            public:
                GPUKernelArgs(const std::shared_ptr<GPUHostParameters>& params);
                GPUKernelArgs(const GPUKernelArgs& args);

                //
                // Add a placeholder parameter for a tensor pointer which will be resolved at runtime.
                //
                GPUKernelArgs& add_placeholder(const std::string& type, const std::string& name);

                //
                // Add a POD argument to the kernel signature and argument list.
                //
                template <typename T>
                typename std::enable_if<!is_container<T>::value, GPUKernelArgs&>::type
                    add(const std::string& name, const T& arg)
                {
                    return add_argument(name, arg);
                }

                //
                // Add POD arguments as above, but by expanding the array arguments and
                // and adding each individual arg as kernel register arguments.
                //
                template <typename T>
                typename std::enable_if<is_container<T>::value, GPUKernelArgs&>::type
                    add(const std::string& name, const T& arg)
                {
                    return add_arguments(name, arg);
                }

                //
                // Retrieve the kernel argument list for use with the launch primitive.
                //
                void** get_argument_list() { return m_argument_list.data(); }
                //
                // Replace placeholder argument with specifed address.
                //
                GPUKernelArgs& resolve_placeholder(size_t arg_num, void* address);

                //
                // Retrieve the kernel parameter signature given the added kernel arguments.
                //
                std::string get_input_signature();
                size_t get_size() { return m_argument_list.size(); }
            private:
                //
                // Cache the host argument for persistence, add it to the argument list,
                // and add its signature to the kernel input signature.
                //
                template <typename T>
                GPUKernelArgs& add_argument(const std::string& name, const T& arg)
                {
                    validate();
                    void* host_arg = m_host_parameters->cache(arg);
                    m_argument_list.push_back(host_arg);
                    m_placeholder_positions.push_back(false);
                    add_to_signature(type_names.at(std::type_index(typeid(T))), name);
                    return *this;
                }

                //
                // Same as above for a container type T.
                //
                template <typename T>
                GPUKernelArgs& add_arguments(const std::string& name, const T& args)
                {
                    validate();

                    size_t i = 0;
                    for (auto const& arg : args)
                    {
                        add_argument(name + std::to_string(i++), arg);
                    }
                    return *this;
                }

                void validate();

                //
                // Given an input argument type and name, add it to the kernel parameter signature.
                //
                void add_to_signature(const std::string& type, const std::string& name);

            private:
                bool m_signature_generated;
                std::vector<void*> m_argument_list;
                std::vector<bool> m_placeholder_positions;
                std::stringstream m_input_signature;
                std::shared_ptr<GPUHostParameters> m_host_parameters;
                static const std::unordered_map<std::type_index, std::string> type_names;
            };

            //
            // Helper structs to deduce whether a type is iterable.
            //
            template <typename T>
            struct has_const_iterator
            {
            private:
                typedef struct
                {
                    char x;
                } true_type;
                typedef struct
                {
                    char x[2];
                } false_type;

                template <typename U>
                static true_type check(typename U::const_iterator*);
                template <typename U>
                static false_type check(...);

            public:
                static const bool value = sizeof(check<T>(0)) == sizeof(true_type);
                typedef T type;
            };

            template <typename T>
            struct is_container : std::integral_constant<bool, has_const_iterator<T>::value>
            {
            };
        }
    }
}
