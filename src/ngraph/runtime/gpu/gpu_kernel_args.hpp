/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

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
            // Helper structs to deduce whether a type is iterable
            template <typename T>
            struct has_const_iterator;
            template <typename T>
            struct is_container;

            class GPUKernelArgs
            {
            public:
                GPUKernelArgs(const std::shared_ptr<GPUHostParameters>& params);
                GPUKernelArgs(const GPUKernelArgs& args);

                GPUKernelArgs& add_placeholder(const std::string& type, const std::string& name);

                template <typename T>
                typename std::enable_if<!is_container<T>::value, GPUKernelArgs&>::type
                    add(const std::string& name, const T& arg)
                {
                    add_argument(name, arg);
                }

                template <typename T>
                typename std::enable_if<is_container<T>::value, GPUKernelArgs&>::type
                    add(const std::string& name, const T& arg)
                {
                    add_arguments(name, arg);
                }

                template <typename... Args>
                void** get_argument_list(Args&&... args)
                {
                    size_t num_args = sizeof...(args);
                    void* arg_list[] = {args...};

                    size_t i = 0;
                    for (size_t n = 0; n < m_argument_list.size(); n++)
                    {
                        if (m_placeholder_positions[n])
                        {
                            if (i >= num_args)
                            {
                                throw std::runtime_error("Too few kernel arguments supplied for resolving placeholder parameters.");
                            }
                            m_argument_list[n] = arg_list[i++];
                        }
                    }
                    if (i != num_args)
                    {
                        throw std::runtime_error("Too many kernel arguments supplied for resolving placeholder parameters.");
                    }
                    return m_argument_list.data();
                }

                std::string get_input_signature();

            private:
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
                std::string add_to_signature(const std::string& type, const std::string& name);

            private:
                bool m_signature_generated;
                std::vector<void*> m_argument_list;
                std::vector<bool> m_placeholder_positions;
                std::stringstream m_input_signature;
                std::shared_ptr<GPUHostParameters> m_host_parameters;
                static const std::unordered_map<std::type_index, std::string> type_names;
            };

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
                    char x, y;
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
