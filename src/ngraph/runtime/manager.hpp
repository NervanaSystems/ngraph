/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace ngraph
{
    class Function;

    namespace runtime
    {
        class Backend;
        class ExternalFunction;

        /// @brief Interface to a generic manager.
        ///
        /// A manager provides access to compilation for a backend, and a means to obtain
        /// a backed for execution and allocation.
        class Manager
        {
        public:
            virtual ~Manager() {}
            /// @brief Allocate a backend for this transformer.
            ///
            /// Specific transformers may provide addtional methods for allocating customized backends.
            virtual std::shared_ptr<Backend> allocate_backend() = 0;

            /// @brief Convert a function to a form that can be run on a backend.
            virtual std::shared_ptr<ExternalFunction>
                compile(const std::shared_ptr<ngraph::Function>& fun) = 0;

            using Factory = std::function<std::shared_ptr<Manager>(const std::string&)>;

            static std::shared_ptr<Manager> get(const std::string& name);

            static Factory register_factory(const std::string& name, Factory factory);

        private:
            using FactoryMap = std::map<std::string, Factory>;

            static FactoryMap& get_factory_map();
        };
    }
}
