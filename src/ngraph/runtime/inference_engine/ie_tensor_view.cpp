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

#include <cstring>
#include <memory>

#include "ie_tensor_view.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

runtime::inference_engine::IETensorView::IETensorView(const element::Type& element_type,
                                                      const PartialShape& shape)
    : runtime::Tensor(std::make_shared<descriptor::Tensor>(element_type, shape, ""))
{
    m_descriptor->set_tensor_layout(
        std::make_shared<descriptor::layout::DenseTensorLayout>(*m_descriptor));
}

runtime::inference_engine::IETensorView::IETensorView(const element::Type& element_type,
                                                      const Shape& shape)
    : runtime::Tensor(std::make_shared<descriptor::Tensor>(element_type, shape, ""))
{
    m_descriptor->set_tensor_layout(
        std::make_shared<descriptor::layout::DenseTensorLayout>(*m_descriptor));
}

void runtime::inference_engine::IETensorView::write(const void* p, size_t n)
{
    const int8_t* v = (const int8_t*)p;
    if (v == nullptr)
        return;
    data.resize(n);
    std::copy(v, v + n, data.begin());
}

void runtime::inference_engine::IETensorView::read(void* p, size_t n) const
{
    int8_t* v = (int8_t*)p;
    if (v == nullptr)
        return;
    if (n > data.size())
        n = data.size();
    std::copy(data.begin(), data.begin() + n, v);
}
