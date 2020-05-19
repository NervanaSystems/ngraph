//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph
{
    namespace test
    {
        class ValueHolder
        {
            template <typename T>
            T& invalid()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }

        public:
            virtual ~ValueHolder() {}
            virtual operator bool&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator float&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator double&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::string&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator int8_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator int16_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator int32_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator int64_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator uint8_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator uint16_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator uint32_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator uint64_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<std::string>&()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }
            virtual operator std::vector<float>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<double>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<int8_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<int16_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<int32_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<int64_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<uint8_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<uint16_t>&()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }
            virtual operator std::vector<uint32_t>&()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }
            virtual operator std::vector<uint64_t>&()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }
            virtual operator HostTensorPtr&() { NGRAPH_CHECK(false, "Invalid type access"); }
        };

        template <typename T>
        class ValueHolderImp : public ValueHolder
        {
        public:
            ValueHolderImp(const T& value)
                : m_value(value)
            {
            }
            operator T&() override { return m_value; }
        protected:
            T m_value;
        };

        class ValueMap
        {
            using map_type = std::unordered_map<std::string, std::shared_ptr<ValueHolder>>;

        public:
            template <typename T>
            void insert(const std::string& name, const T& value)
            {
                std::pair<map_type::iterator, bool> result = m_values.insert(
                    map_type::value_type(name, std::make_shared<ValueHolderImp<T>>(value)));
                NGRAPH_CHECK(result.second, name, " is already in use");
            }
            template <typename T>
            T& get(const std::string& name)
            {
                return static_cast<T&>(*m_values.at(name));
            }

        protected:
            map_type m_values;
        };

        class DeserializeAttributeVisitor : public AttributeVisitor
        {
        public:
            DeserializeAttributeVisitor(ValueMap& value_map)
                : m_values(value_map)
            {
            }
            void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override
            {
                NGRAPH_CHECK(false, "Attribute \"", name, "\" cannot be unmarshalled");
            }
            // The remaining adapter methods fall back on the void adapter if not implemented
            void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override
            {
                adapter.set(m_values.get<std::string>(name));
            };
            void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override
            {
                adapter.set(m_values.get<bool>(name));
            };
            void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override
            {
                adapter.set(m_values.get<int64_t>(name));
            }
            void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override
            {
                adapter.set(m_values.get<double>(name));
            }

            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int8_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<int8_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int16_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<int16_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int32_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<int32_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int64_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<int64_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint8_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<uint8_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint16_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<uint16_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint32_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<uint32_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint64_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<uint64_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<std::string>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<std::string>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<float>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<float>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<double>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<double>>(name));
            }
            void on_adapter(const std::string& name, ValueAccessor<void*>& adapter) override
            {
                HostTensorPtr& data = m_values.get<HostTensorPtr>(name);
                data->read(adapter.get_ptr(), adapter.size());
            }

        protected:
            ValueMap& m_values;
        };

        class SerializeAttributeVisitor : public AttributeVisitor
        {
        public:
            SerializeAttributeVisitor(ValueMap& value_map)
                : m_values(value_map)
            {
            }

            void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override
            {
                NGRAPH_CHECK(false, "Attribute \"", name, "\" cannot be marshalled");
            }
            // The remaining adapter methods fall back on the void adapter if not implemented
            void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override
            {
                m_values.insert(name, adapter.get());
            };
            void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override
            {
                m_values.insert(name, adapter.get());
            };

            void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<std::string>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<float>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<double>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int8_t>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int16_t>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int32_t>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int64_t>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint8_t>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint16_t>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint32_t>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint64_t>>& adapter) override
            {
                m_values.insert(name, adapter.get());
            }
            void on_adapter(const std::string& name, ValueAccessor<void*>& adapter) override
            {
                HostTensorPtr data =
                    std::make_shared<HostTensor>(element::u8, Shape{adapter.size()});
                data->write(adapter.get_ptr(), adapter.size());
                m_values.insert(name, data);
            }

        protected:
            ValueMap& m_values;
        };

        class NodeBuilder : public ValueMap, public DeserializeAttributeVisitor
        {
        public:
            NodeBuilder()
                : DeserializeAttributeVisitor(static_cast<ValueMap&>(*this))
                , m_serializer(*this)
            {
            }

            NodeBuilder(const std::shared_ptr<Node>& node)
                : DeserializeAttributeVisitor(static_cast<ValueMap&>(*this))
                , m_serializer(*this)
            {
                save_node(node);
            }

            void save_node(std::shared_ptr<Node> node)
            {
                m_node_type_info = node->get_type_info();
                node->visit_attributes(m_serializer);
            }

            // Does not validate, since inputs aren't set
            std::shared_ptr<Node> create()
            {
                std::shared_ptr<Node> node(FactoryRegistry<Node>::get().create(m_node_type_info));
                node->visit_attributes(*this);
                return node;
            }
            AttributeVisitor& get_node_saver() { return m_serializer; }
            AttributeVisitor& get_node_loader() { return *this; }
        protected:
            Node::type_info_t m_node_type_info;
            SerializeAttributeVisitor m_serializer;
        };
    }
}
