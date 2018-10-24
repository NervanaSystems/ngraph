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

#include <fstream>

#include "ngraph/except.hpp"

#include "core/graph.hpp"
#include "core/model.hpp"
#include "core/node.hpp"

#include "onnx.hpp"
#include "ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            namespace error
            {
                struct file_open : ngraph_error
                {
                    explicit file_open(const std::string& path)
                        : ngraph_error{"failure opening file:" + path}
                    {
                    }
                };

                struct stream_parse : ngraph_error
                {
                    explicit stream_parse(std::istream&)
                        : ngraph_error{"failure parsing data from the stream"}
                    {
                    }
                };

            } // namespace error

        } // namespace detail

        struct Weight::Impl
        {
        public:
            Impl(const Impl&) = default;
            Impl& operator=(const Impl&) = delete;

            Impl() = delete;

            Impl(Impl&&) = default;
            Impl& operator=(Impl&&) = delete;

            Impl(const element::Type& type, const Shape& shape, std::vector<char> data)
                : m_shape{shape},
                  m_type{type},
                  m_data{std::move(data)}
            {
                for (const auto& value : m_shape)
                {
                    m_size *= value;
                }
            }

            Impl(Type type, std::size_t dimensions, const std::size_t* shape, const void* data)
                : Impl{to_element_type(type), {shape, shape + dimensions},
                       {reinterpret_cast<const char*>(data), reinterpret_cast<const char*>(data) + dimensions}}
            {
            }

            const Shape& shape() const
            {
                return m_shape;
            }

            std::size_t size() const
            {
                return m_size;
            }

            const element::Type& type() const
            {
                return m_type;
            }

            const void* data() const
            {
                return reinterpret_cast<const void*>(m_data.data());
            }

        private:
            Shape m_shape{};
            const element::Type& m_type;
            std::size_t m_size{1};
            std::vector<char> m_data{};

            const element::Type& to_element_type(Type type)
            {
                switch (type)
                {
                case Type::f16:
                case Type::f32:
                    return element::f32;
                case Type::f64:
                    return element::f64;
                case Type::i8:
                    return element::i8;
                case Type::i16:
                    return element::i16;
                case Type::i32:
                    return element::i32;
                case Type::i64:
                    return element::i64;
                case Type::u8:
                    return element::u8;
                case Type::u16:
                    return element::u16;
                case Type::u32:
                    return element::u32;
                case Type::u64:
                    return element::u64;
                }
            }
        };

        Weight::Weight(Type type, std::size_t dimensions, const std::size_t* shape, const void* data)
            : m_pimpl{new Impl{type, dimensions, shape, data}, [](Impl* impl) { delete impl; }}
        {
        }

        Weight::Weight(const Weight& other)
            : m_pimpl{new Impl{*other.m_pimpl}, [](Impl* impl) { delete impl; }}
        {
        }

        Weight& Weight::operator=(const Weight& other)
        {
            if (this != &other)
            {

            }
            return *this;
        }

        const element::Type& Weight::type() const
        {
            return m_pimpl->type();
        }

        const void* Weight::data() const
        {
            return m_pimpl->data();
        }

        const Shape& Weight::shape() const
        {
            return m_pimpl->shape();
        }

        std::vector<std::shared_ptr<Function>> load_onnx_model(std::istream& sin,
                const Weights& weights)
        {
            onnx::ModelProto model_proto;
            if (!model_proto.ParseFromIstream(&sin))
            {
                throw detail::error::stream_parse{sin};
            }
            std::vector<std::shared_ptr<Function>> output_functions;
            Model model{model_proto};
            Graph graph{model_proto.graph(), model, weights};
            for (const auto& output : graph.get_outputs())
            {
                output_functions.emplace_back(std::make_shared<Function>(
                    graph.get_ng_node_from_cache(output.get_name()), graph.get_ng_parameters()));
            }
            return output_functions;
        }

        std::vector<std::shared_ptr<Function>> load_onnx_model(const std::string& path,
                const Weights& weights)
        {
            std::ifstream ifs{path, std::ios::in | std::ios::binary};
            if (!ifs.is_open())
            {
                throw detail::error::file_open{path};
            }
            return load_onnx_model(ifs, weights);
        }

        std::shared_ptr<Function> import_onnx_function(std::istream& sin,
                const Weights& weights)
        {
            return load_onnx_model(sin, weights).front();
        }

        std::shared_ptr<Function> import_onnx_function(const std::string& path,
                const Weights& weights)
        {
            return load_onnx_model(path, weights).front();
        }

        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn)
        {
            OperatorsBridge::register_operator(name, version, domain, std::move(fn));
        }

    } // namespace onnx_import

} // namespace ngraph
