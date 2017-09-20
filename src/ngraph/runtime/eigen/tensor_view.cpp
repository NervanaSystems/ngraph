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

#include <Eigen/Dense>
#include <memory>

#include "ngraph/ngraph.hpp"

using namespace Eigen;
using namespace ngraph::runtime::eigen;
using namespace ngraph::element;

template void ngraph::runtime::eigen::add<Float32>(const PrimaryTensorView<Float32>& arg0,
                                                   const PrimaryTensorView<Float32>& arg1,
                                                   PrimaryTensorView<Float32>&       out);

template void ngraph::runtime::eigen::multiply<Float32>(const PrimaryTensorView<Float32>& arg0,
                                                        const PrimaryTensorView<Float32>& arg1,
                                                        PrimaryTensorView<Float32>&       out);

std::shared_ptr<ngraph::runtime::PrimaryTensorView> ngraph::runtime::eigen::make_tensor_view(
    std::shared_ptr<ngraph::descriptor::TensorView> tensor_view)
{
    // For now, we only support Float32 primary tensor views
    return std::make_shared<PrimaryTensorView<Float32>>(
        tensor_view->get_tensor_view_type()->get_shape());
}
