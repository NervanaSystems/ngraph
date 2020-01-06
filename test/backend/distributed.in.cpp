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

#include <fstream>
#include <sstream>

#include "gtest/gtest.h"

#include "ngraph/distributed.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close_f.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

static void test_allreduce_common(reduction::Type reduce_type)
{
    auto comm_size = get_distributed_interface()->get_size();
    if (comm_size > 1)
    {
        auto shape = Shape{2, 2};
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto f =
            make_shared<Function>(make_shared<op::AllReduce>(A, reduce_type), ParameterVector{A});

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        auto v = vector<float>{1, 2, 3, 4};
        auto a = backend->create_tensor(element::f32, shape);
        auto result = backend->create_tensor(element::f32, shape);

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
        switch (reduce_type)
        {
        case reduction::Type::SUM:
            copy_data(a, v);
            std::transform(v.begin(), v.end(), v.begin(), [=](float x) { return x * comm_size; });
            break;
        case reduction::Type::PROD:
            copy_data(a, v);
            std::transform(v.begin(), v.end(), v.begin(), [&](float elm) -> float {
                return pow(elm, comm_size);
            });
            break;
        case reduction::Type::MIN:
        case reduction::Type::MAX:
            auto shift = get_distributed_interface()->get_rank();
            std::rotate(v.begin(), v.begin() + shift % v.size(), v.end());
            copy_data(a, v);
            if (reduce_type == reduction::Type::MIN)
            {
                std::fill(v.begin(), v.end(), 1);
                for (int i = 1; i < static_cast<int>(v.size()) - comm_size + 1; i++)
                    v[i] = i + 1;
            }
            else
            {
                std::fill(v.begin(), v.end(), v.size());
                for (int i = 0; i < static_cast<int>(v.size()) - comm_size; i++)
                    v[i] = i + 2;
            }
        }
#pragma GCC diagnostic pop

        auto handle = backend->compile(f);
        handle->call_with_validate({result}, {a});
        EXPECT_TRUE(test::all_close_f(v, read_vector<float>(result)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, allreduce_sum)
{
    test_allreduce_common(reduction::Type::SUM);
}

NGRAPH_TEST(${BACKEND_NAME}, allreduce_min)
{
    test_allreduce_common(reduction::Type::MIN);
}

NGRAPH_TEST(${BACKEND_NAME}, allreduce_max)
{
    test_allreduce_common(reduction::Type::MAX);
}

#if !defined(NGRAPH_DISTRIBUTED_MLSL_ENABLE)
NGRAPH_TEST(${BACKEND_NAME}, allreduce_prod)
{
    test_allreduce_common(reduction::Type::PROD);
}
#endif

NGRAPH_TEST(${BACKEND_NAME}, broadcastdistributed)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto comm_size = get_distributed_interface()->get_size();
    for (int root_id = 0; root_id < comm_size; ++root_id)
    {
        auto f = make_shared<Function>(make_shared<op::BroadcastDistributed>(A, root_id),
                                       ParameterVector{A});

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        auto v = vector<float>{1, 2, 3, 4};
        auto result = backend->create_tensor(element::f32, shape);
        copy_data(result, vector<float>(4, 0));

        auto processIdx = get_distributed_interface()->get_rank();
        if (processIdx == root_id)
        {
            copy_data(result, v);
        }

        auto handle = backend->compile(f);
        handle->call_with_validate({result}, {result});
        EXPECT_EQ(v, read_vector<float>(result));
    }
}

// MLSL does not support send recv
#if !defined(NGRAPH_DISTRIBUTED_MLSL_ENABLE)
NGRAPH_TEST(${BACKEND_NAME}, send_recv)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto comm_size = get_distributed_interface()->get_size();
    // this test only works for 2 nodes
    if (comm_size != 2)
    {
        return;
    }
    auto rank = get_distributed_interface()->get_rank();
    std::shared_ptr<Function> f;
    if (rank == 0)
    {
        f = make_shared<Function>(make_shared<op::Send>(A, 1), ParameterVector{A});
    }
    else
    {
        f = make_shared<Function>(make_shared<op::Recv>(A, 0), ParameterVector{A});
    }
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto v = vector<float>{1, 2, 3, 4};
    auto result = backend->create_tensor(element::f32, shape);
    copy_data(result, vector<float>(4, 0));

    if (rank == 0)
    {
        copy_data(result, v);
    }

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {result});
    EXPECT_EQ(v, read_vector<float>(result));
}
#endif

// MLSL does not support send recv
#if !defined(NGRAPH_DISTRIBUTED_MLSL_ENABLE)
NGRAPH_TEST(${BACKEND_NAME}, send_recv_ring)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto comm_size = get_distributed_interface()->get_size();
    // test only works for at least 2 nodes
    if (comm_size < 2)
    {
        return;
    }

    auto rank = get_distributed_interface()->get_rank();
    std::shared_ptr<Function> f_send;
    std::shared_ptr<Function> f_recv;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto v = vector<float>{1, 2, 3, 4};
    auto result = backend->create_tensor(element::f32, shape);
    copy_data(result, vector<float>(4, 0));

    if (rank != 0)
    {
        f_recv = make_shared<Function>(make_shared<op::Recv>(A, rank - 1), ParameterVector{A});
        auto handle = backend->compile(f_recv);
        handle->call_with_validate({result}, {result});
        EXPECT_EQ(v, read_vector<float>(result));
    }
    else
    {
        copy_data(result, v);
    }

    f_send =
        make_shared<Function>(make_shared<op::Send>(A, (rank + 1) % comm_size), ParameterVector{A});
    backend->compile(f_send)->call_with_validate({result}, {result});

    if (rank == 0)
    {
        f_recv = make_shared<Function>(make_shared<op::Recv>(A, comm_size - 1), ParameterVector{A});
        auto handle = backend->compile(f_recv);
        copy_data(result, vector<float>(4, 0));
        backend->compile(f_recv)->call_with_validate({result}, {result});
        EXPECT_EQ(v, read_vector<float>(result));
    }
}
#endif
