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
#include "ngraph/log.hpp"

using namespace ngraph;
using namespace std;

class Indexer_0 : public runtime::opt_kernel::ReshapeIndexer::Indexer
{
public:
    Indexer_0(const Shape& in_shape, const AxisVector& in_axis_order)
        : Indexer()
    {
    }
    size_t next() override
    {
        NGRAPH_INFO;
        return 0;
    }
};

class Indexer_1 : public runtime::opt_kernel::ReshapeIndexer::Indexer
{
public:
    Indexer_1(const Shape& in_shape, const AxisVector& in_axis_order)
        : Indexer()
        , m_in_shape{in_shape}
    {
        for (size_t i = 0; i < 1; i++)
        {
            m_size[i] = in_shape[in_axis_order[i]];
            m_in_index[i] = 0;
            m_map_index[in_axis_order[i]] = &m_in_index[i];
        }
    }
    size_t next() override
    {
        size_t rc = *m_map_index[0];
        m_in_index[0]++;
        return rc;
    }

private:
    const Shape& m_in_shape;
    size_t m_size[1];
    size_t m_in_index[1];
    size_t* m_map_index[1];
};

class Indexer_2 : public runtime::opt_kernel::ReshapeIndexer::Indexer
{
public:
    Indexer_2(const Shape& in_shape, const AxisVector& in_axis_order)
        : Indexer()
        , m_in_shape{in_shape}
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

class Indexer_3 : public runtime::opt_kernel::ReshapeIndexer::Indexer
{
public:
    Indexer_3(const Shape& in_shape, const AxisVector& in_axis_order)
        : Indexer()
        , m_in_shape{in_shape}
    {
        for (size_t i = 0; i < 3; i++)
        {
            m_size[i] = in_shape[in_axis_order[i]];
            m_in_index[i] = 0;
            m_map_index[in_axis_order[i]] = &m_in_index[i];
        }
    }
    size_t next() override
    {
        // clang-format off
        size_t rc = *m_map_index[0] * m_in_shape[1] * m_in_shape[2]
                  + *m_map_index[1] * m_in_shape[2]
                  + *m_map_index[2];
        // clang-format on
        m_in_index[2]++;
        if (m_in_index[2] == m_size[2])
        {
            m_in_index[2] = 0;
            m_in_index[1]++;
            if (m_in_index[1] == m_size[1])
            {
                m_in_index[1] = 0;
                m_in_index[0]++;
            }
        }
        return rc;
    }

private:
    const Shape& m_in_shape;
    size_t m_size[3];
    size_t m_in_index[3];
    size_t* m_map_index[3];
};

class Indexer_4 : public runtime::opt_kernel::ReshapeIndexer::Indexer
{
public:
    Indexer_4(const Shape& in_shape, const AxisVector& in_axis_order)
        : Indexer()
        , m_in_shape{in_shape}
    {
        for (size_t i = 0; i < 4; i++)
        {
            m_size[i] = in_shape[in_axis_order[i]];
            m_in_index[i] = 0;
            m_map_index[in_axis_order[i]] = &m_in_index[i];
        }
    }
    size_t next() override
    {
        // clang-format off
        size_t rc = *m_map_index[0] * m_in_shape[1] * m_in_shape[2] * m_in_shape[3]
                  + *m_map_index[1] * m_in_shape[2] * m_in_shape[3]
                  + *m_map_index[2] * m_in_shape[3]
                  + *m_map_index[3];
        // clang-format on
        m_in_index[3]++;
        if (m_in_index[3] == m_size[3])
        {
            m_in_index[3] = 0;
            m_in_index[2]++;
            if (m_in_index[2] == m_size[2])
            {
                m_in_index[2] = 0;
                m_in_index[1]++;
                if (m_in_index[1] == m_size[1])
                {
                    m_in_index[1] = 0;
                    m_in_index[0]++;
                }
            }
        }
        return rc;
    }

private:
    const Shape& m_in_shape;
    size_t m_size[4];
    size_t m_in_index[4];
    size_t* m_map_index[4];
};

class Indexer_5 : public runtime::opt_kernel::ReshapeIndexer::Indexer
{
public:
    Indexer_5(const Shape& in_shape, const AxisVector& in_axis_order)
        : Indexer()
        , m_in_shape{in_shape}
    {
        for (size_t i = 0; i < 5; i++)
        {
            m_size[i] = in_shape[in_axis_order[i]];
            m_in_index[i] = 0;
            m_map_index[in_axis_order[i]] = &m_in_index[i];
        }
    }
    size_t next() override
    {
        // clang-format off
        size_t rc = *m_map_index[0] * m_in_shape[1] * m_in_shape[2] * m_in_shape[3] * m_in_shape[4]
                  + *m_map_index[1] * m_in_shape[2] * m_in_shape[3] * m_in_shape[4]
                  + *m_map_index[2] * m_in_shape[3] * m_in_shape[4]
                  + *m_map_index[3] * m_in_shape[4]
                  + *m_map_index[4];
        // clang-format on
        m_in_index[4]++;
        if (m_in_index[4] == m_size[4])
        {
            m_in_index[4] = 0;
            m_in_index[3]++;
            if (m_in_index[3] == m_size[3])
            {
                m_in_index[3] = 0;
                m_in_index[2]++;
                if (m_in_index[2] == m_size[2])
                {
                    m_in_index[2] = 0;
                    m_in_index[1]++;
                    if (m_in_index[1] == m_size[1])
                    {
                        m_in_index[1] = 0;
                        m_in_index[0]++;
                    }
                }
            }
        }
        return rc;
    }

private:
    const Shape& m_in_shape;
    size_t m_size[5];
    size_t m_in_index[5];
    size_t* m_map_index[5];
};

class Indexer_6 : public runtime::opt_kernel::ReshapeIndexer::Indexer
{
public:
    Indexer_6(const Shape& in_shape, const AxisVector& in_axis_order)
        : Indexer()
        , m_in_shape{in_shape}
    {
        for (size_t i = 0; i < 6; i++)
        {
            m_size[i] = in_shape[in_axis_order[i]];
            m_in_index[i] = 0;
            m_map_index[in_axis_order[i]] = &m_in_index[i];
        }
    }
    size_t next() override
    {
        // clang-format off
        size_t rc = *m_map_index[0] * m_in_shape[1] * m_in_shape[2] * m_in_shape[3] * m_in_shape[4] * m_in_shape[5]
                  + *m_map_index[1] * m_in_shape[2] * m_in_shape[3] * m_in_shape[4] * m_in_shape[5]
                  + *m_map_index[2] * m_in_shape[3] * m_in_shape[4] * m_in_shape[5]
                  + *m_map_index[3] * m_in_shape[4] * m_in_shape[5]
                  + *m_map_index[4] * m_in_shape[5]
                  + *m_map_index[5];
        // clang-format on
        m_in_index[5]++;
        if (m_in_index[5] == m_size[5])
        {
            m_in_index[5] = 0;
            m_in_index[4]++;
            if (m_in_index[4] == m_size[4])
            {
                m_in_index[4] = 0;
                m_in_index[3]++;
                if (m_in_index[3] == m_size[3])
                {
                    m_in_index[3] = 0;
                    m_in_index[2]++;
                    if (m_in_index[2] == m_size[2])
                    {
                        m_in_index[2] = 0;
                        m_in_index[1]++;
                        if (m_in_index[1] == m_size[1])
                        {
                            m_in_index[1] = 0;
                            m_in_index[0]++;
                        }
                    }
                }
            }
        }
        return rc;
    }

private:
    const Shape& m_in_shape;
    size_t m_size[6];
    size_t m_in_index[6];
    size_t* m_map_index[6];
};

runtime::opt_kernel::ReshapeIndexer::ReshapeIndexer(const Shape& in_shape,
                                                    const AxisVector& in_axis_order)
{
    NGRAPH_INFO << in_shape.size();
    switch (in_shape.size())
    {
    case 0: m_indexer.reset(new Indexer_0(in_shape, in_axis_order)); break;
    case 1: m_indexer.reset(new Indexer_1(in_shape, in_axis_order)); break;
    case 2: m_indexer.reset(new Indexer_2(in_shape, in_axis_order)); break;
    case 3: m_indexer.reset(new Indexer_3(in_shape, in_axis_order)); break;
    case 4: m_indexer.reset(new Indexer_4(in_shape, in_axis_order)); break;
    case 5: m_indexer.reset(new Indexer_5(in_shape, in_axis_order)); break;
    case 6: m_indexer.reset(new Indexer_6(in_shape, in_axis_order)); break;
    default: throw runtime_error("Unsupported dimention in ReshapeIndexer");
    }
    NGRAPH_INFO;
}

size_t runtime::opt_kernel::ReshapeIndexer::next()
{
    return m_indexer->next();
}
