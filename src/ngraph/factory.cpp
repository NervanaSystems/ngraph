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

#include <mutex>

#include "ngraph/factory.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast_distributed.hpp"
#include "ngraph/op/parameter.hpp"

using namespace std;

namespace ngraph
{
    mutex& get_registry_mutex()
    {
        static mutex registry_mutex;
        return registry_mutex;
    }

    template <>
    FactoryRegistry<Node>& FactoryRegistry<Node>::get()
    {
        static FactoryRegistry<Node> registry;
        static mutex init_guard;
        // TODO: Add a lock
        if (registry.m_factory_map.size() == 0)
        {
            lock_guard<mutex> guard(init_guard);
            if (registry.m_factory_map.size() == 0)
            {
                registry.register_factory<op::Abs>();
                registry.register_factory<op::Acos>();
                registry.register_factory<op::Add>();
                registry.register_factory<op::All>();
                registry.register_factory<op::AllReduce>();
                registry.register_factory<op::And>();
                registry.register_factory<op::Any>();
                registry.register_factory<op::ArgMax>();
                registry.register_factory<op::ArgMin>();
                registry.register_factory<op::v0::AvgPool>();
                registry.register_factory<op::v0::AvgPoolBackprop>();
                registry.register_factory<op::v1::AvgPool>();
                registry.register_factory<op::v1::AvgPoolBackprop>();
                registry.register_factory<op::BatchNormInference>();
                registry.register_factory<op::BatchNormTraining>();
                registry.register_factory<op::BatchNormTrainingBackprop>();
                registry.register_factory<op::BroadcastDistributed>();
                registry.register_factory<op::v0::Broadcast>();
                registry.register_factory<op::v0::BroadcastLike>();
                registry.register_factory<op::v1::Broadcast>();
                registry.register_factory<op::Parameter>();
            }
        }
        return registry;
    }
}
