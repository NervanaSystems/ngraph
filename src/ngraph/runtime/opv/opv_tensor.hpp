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


#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"


using namespace ngraph;
using namespace std;


namespace ngraph
{
    namespace runtime
    {
        namespace opv
        {
            class OPVTensor;
        }
    }
}


class ngraph::runtime::opv::OPVTensor : public ngraph::runtime::Tensor
{
public:
    OPVTensor(const ngraph::element::Type& element_type, const Shape& shape);
    OPVTensor(const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer);
    virtual ~OPVTensor() override;

    /// \brief Write bytes directly into the tensor
    /// \param p Pointer to source of data
    /// \param n_bytes Number of bytes to write, must be integral number of elements.
    void write(const void* p, size_t n_bytes) override;

    /// \brief Read bytes directly from the tensor
    /// \param p Pointer to destination for data
    /// \param n_bytes Number of bytes to read, must be integral number of elements.
    void read(void* p, size_t n_bytes) const override;


private:
    OPVTensor(const OPVTensor&) = delete;
    OPVTensor(OPVTensor&&) = delete;
    OPVTensor& operator=(const OPVTensor&) = delete;
    std::vector<int8_t> m_data;
};