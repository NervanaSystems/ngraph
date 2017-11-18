// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/common.hpp"
#include "ngraph/types/element_type.hpp"
#include "ngraph/types/type.hpp"

using namespace ngraph::runtime;

std::shared_ptr<const ngraph::descriptor::TensorView> TensorView::get_tensor_view_descriptor() const
{
    return m_descriptor;
}

std::shared_ptr<ngraph::descriptor::Value> TensorView::get_descriptor() const
{
    return m_descriptor;
}

void TensorView::collect_tensor_views(std::vector<std::shared_ptr<TensorView>>& views,
                                      const std::shared_ptr<Value>& value) const
{
    views.push_back(std::static_pointer_cast<TensorView>(value));
}

const ngraph::Shape& TensorView::get_shape() const
{
    return m_descriptor->get_tensor_view_type()->get_shape();
}

std::shared_ptr<ngraph::descriptor::layout::TensorViewLayout>
    TensorView::get_tensor_view_layout() const
{
    return m_descriptor->get_tensor_view_layout();
}
