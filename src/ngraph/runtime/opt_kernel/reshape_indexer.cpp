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

#include "ngraph/runtime/opt_kernel/reshape_indexer.hpp"

using namespace ngraph;
using namespace std;

// void runtime::opt_kernel::ReshapeIndexer::indexer_0(const Shape& in_shape,
//                                                     const AxisVector& in_axis_order,
//                                                     const Shape& out_shape)
// {
//     *out = *in;
// }

// void runtime::opt_kernel::ReshapeIndexer::indexer_1(const Shape& in_shape,
//                                                     const AxisVector& in_axis_order,
//                                                     const Shape& out_shape)
// {
//     size_t size[1];
//     size_t in_index[1];
//     size_t* map_index[1];
//     for (size_t i = 0; i < 1; i++)
//     {
//         size[i] = in_shape[in_axis_order[i]];
//         map_index[in_axis_order[i]] = &in_index[i];
//     }
//     for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
//     {
//         *out++ = in[*map_index[0]];
//     }
// }

class Indexer_2 : public runtime::opt_kernel::ReshapeIndexer::Indexer
{
public:
    Indexer_2(const Shape& in_shape, const AxisVector& in_axis_order)
        : Indexer(), m_in_shape{in_shape}
    {
        for (size_t i = 0; i < 2; i++)
        {
            m_size[i] = in_shape[in_axis_order[i]];
            m_in_index[i] = 0;
            m_map_index[in_axis_order[i]] = &m_in_index[i];
        }
    }
    size_t next() override
    {
        size_t rc = *m_map_index[0] * m_in_shape[1] + *m_map_index[1];
        m_in_index[1]++;
        if (m_in_index[1] == m_size[1])
        {
            m_in_index[1] = 0;
            m_in_index[0]++;
        }
        return rc;
    }

private:
    const Shape& m_in_shape;
    size_t m_size[2];
    size_t m_in_index[2];
    size_t* m_map_index[2];
};

// void runtime::opt_kernel::ReshapeIndexer::indexer_2(const Shape& in_shape,
//                                                     const AxisVector& in_axis_order,
//                                                     const Shape& out_shape)
// {
//     size_t size[2];
//     size_t in_index[2];
//     size_t* map_index[2];
//     for (size_t i = 0; i < 2; i++)
//     {
//         size[i] = in_shape[in_axis_order[i]];
//         map_index[in_axis_order[i]] = &in_index[i];
//     }
//     for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
//     {
//         for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
//         {
//             *out++ = in[*map_index[0] * in_shape[1] + *map_index[1]];
//         }
//     }
// }

// void runtime::opt_kernel::ReshapeIndexer::indexer_3(const Shape& in_shape,
//                                                     const AxisVector& in_axis_order,
//                                                     const Shape& out_shape)
// {
//     size_t size[3];
//     size_t in_index[3];
//     size_t* map_index[3];
//     for (size_t i = 0; i < 3; i++)
//     {
//         size[i] = in_shape[in_axis_order[i]];
//         map_index[in_axis_order[i]] = &in_index[i];
//     }
//     for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
//     {
//         for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
//         {
//             for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2])
//             {
//                 *out++ = in[*map_index[0] * in_shape[1] * in_shape[2] +
//                             *map_index[1] * in_shape[2] + *map_index[2]];
//             }
//         }
//     }
// }

// void runtime::opt_kernel::ReshapeIndexer::indexer_4(const Shape& in_shape,
//                                                     const AxisVector& in_axis_order,
//                                                     const Shape& out_shape)
// {
//     size_t size[4];
//     size_t in_index[4];
//     size_t* map_index[4];
//     for (size_t i = 0; i < 4; i++)
//     {
//         size[i] = in_shape[in_axis_order[i]];
//         map_index[in_axis_order[i]] = &in_index[i];
//     }
//     for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
//     {
//         for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
//         {
//             for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2])
//             {
//                 for (in_index[3] = 0; in_index[3] < size[3]; ++in_index[3])
//                 {
//                     *out++ = in[*map_index[0] * in_shape[1] * in_shape[2] * in_shape[3] +
//                                 *map_index[1] * in_shape[2] * in_shape[3] +
//                                 *map_index[2] * in_shape[3] + *map_index[3]];
//                 }
//             }
//         }
//     }
// }

// void runtime::opt_kernel::ReshapeIndexer::indexer_5(const Shape& in_shape,
//                                                     const AxisVector& in_axis_order,
//                                                     const Shape& out_shape)
// {
//     size_t size[5];
//     size_t in_index[5];
//     size_t* map_index[5];
//     for (size_t i = 0; i < 5; i++)
//     {
//         size[i] = in_shape[in_axis_order[i]];
//         map_index[in_axis_order[i]] = &in_index[i];
//     }
//     for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
//     {
//         for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
//         {
//             for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2])
//             {
//                 for (in_index[3] = 0; in_index[3] < size[3]; ++in_index[3])
//                 {
//                     for (in_index[4] = 0; in_index[4] < size[4]; ++in_index[4])
//                     {
//                         *out++ = in[*map_index[0] * in_shape[1] * in_shape[2] * in_shape[3] *
//                                         in_shape[4] +
//                                     *map_index[1] * in_shape[2] * in_shape[3] * in_shape[4] +
//                                     *map_index[2] * in_shape[3] * in_shape[4] +
//                                     *map_index[3] * in_shape[4] + *map_index[4]];
//                     }
//                 }
//             }
//         }
//     }
// }

// void runtime::opt_kernel::ReshapeIndexer::indexer_6(const Shape& in_shape,
//                                                     const AxisVector& in_axis_order,
//                                                     const Shape& out_shape)
// {
//     size_t size[6];
//     size_t in_index[6];
//     size_t* map_index[6];
//     for (size_t i = 0; i < 6; i++)
//     {
//         size[i] = in_shape[in_axis_order[i]];
//         map_index[in_axis_order[i]] = &in_index[i];
//     }
//     for (in_index[0] = 0; in_index[0] < size[0]; ++in_index[0])
//     {
//         for (in_index[1] = 0; in_index[1] < size[1]; ++in_index[1])
//         {
//             for (in_index[2] = 0; in_index[2] < size[2]; ++in_index[2])
//             {
//                 for (in_index[3] = 0; in_index[3] < size[3]; ++in_index[3])
//                 {
//                     for (in_index[4] = 0; in_index[4] < size[4]; ++in_index[4])
//                     {
//                         for (in_index[5] = 0; in_index[5] < size[5]; ++in_index[5])
//                         {
//                             *out++ = in[*map_index[0] * in_shape[1] * in_shape[2] * in_shape[3] *
//                                             in_shape[4] * in_shape[5] +
//                                         *map_index[1] * in_shape[2] * in_shape[3] * in_shape[4] *
//                                             in_shape[5] +
//                                         *map_index[2] * in_shape[3] * in_shape[4] * in_shape[5] +
//                                         *map_index[3] * in_shape[4] * in_shape[5] +
//                                         *map_index[4] * in_shape[5] + *map_index[5]];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

runtime::opt_kernel::ReshapeIndexer::ReshapeIndexer(const Shape& in_shape,
                                                    const AxisVector& in_axis_order,
                                                    const Shape& out_shape)
{
    switch (in_shape.size())
    {
    case 0: break;
    case 1: break;
    case 2: m_indexer.reset(new Indexer_2(in_shape, in_axis_order)); break;
    case 3: break;
    case 4: break;
    case 5: break;
    case 6: break;
    default: throw runtime_error("Unsupported dimention in ReshapeIndexer");
    }
}

size_t runtime::opt_kernel::ReshapeIndexer::next()
{
}
