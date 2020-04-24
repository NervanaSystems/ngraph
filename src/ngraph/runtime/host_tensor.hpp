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

#include <memory>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"

namespace ngraph
{
    namespace runtime
    {
        class HostTensor;
    }
    namespace op
    {
        namespace v0
        {
            class Constant;
        }
    }
}

class NGRAPH_API ngraph::runtime::HostTensor : public ngraph::runtime::Tensor
{
public:
    HostTensor(const ngraph::element::Type& element_type,
               const Shape& shape,
               const std::string& name);
    HostTensor(const ngraph::element::Type& element_type,
               const Shape& shape,
               void* memory_pointer,
               const std::string& name);
    HostTensor(const ngraph::element::Type& element_type, const Shape& shape);
    HostTensor(const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer);
    HostTensor(const std::shared_ptr<op::v0::Constant>& constant);
    virtual ~HostTensor() override;

    void* get_data_ptr();
    const void* get_data_ptr() const;

    template <typename T>
    T* get_data_ptr()
    {
        return static_cast<T*>(get_data_ptr());
    }

    template <typename T>
    const T* get_data_ptr() const
    {
        return static_cast<T*>(get_data_ptr());
    }

    template <element::Type_t ET>
    typename element_type_traits<ET>::value_type* get_data_ptr()
    {
        return static_cast<typename element_type_traits<ET>::value_type*>(get_data_ptr());
    }

    template <element::Type_t ET>
    const typename element_type_traits<ET>::value_type* get_data_ptr() const
    {
        return static_cast<typename element_type_traits<ET>::value_type>(get_data_ptr());
    }

    /// \brief Write bytes directly into the tensor
    /// \param p Pointer to source of data
    /// \param n Number of bytes to write, must be integral number of elements.
    void write(const void* p, size_t n) override;

    /// \brief Read bytes directly from the tensor
    /// \param p Pointer to destination for data
    /// \param n Number of bytes to read, must be integral number of elements.
    void read(void* p, size_t n) const override;

    class NGRAPH_API HostEvaluatorTensor : public EvaluatorTensor
    {
    protected:
        using EvaluatorTensor::EvaluatorTensor;

    public:
        virtual std::shared_ptr<HostTensor> get_host_tensor() = 0;
    };
    using HostEvaluatorTensorPtr = std::shared_ptr<HostEvaluatorTensor>;
    using HostEvaluatorTensorVector = std::vector<HostEvaluatorTensorPtr>;
    /// \brief Get an evaluator tensor that uses this host tensor for data
    static HostEvaluatorTensorPtr create_evaluator_tensor(std::shared_ptr<HostTensor> host_tensor);
    /// \brief Get an evaluator tensor that creates a host tensor on demand
    /// \param element_type Constraint for element type
    /// \param partial_shape Constraint for partial shape
    static HostEvaluatorTensorPtr create_evaluator_tensor(const element::Type& element_type,
                                                          const PartialShape& partial_shape,
                                                          const std::string& name = "");

private:
    HostTensor(const HostTensor&) = delete;
    HostTensor(HostTensor&&) = delete;
    HostTensor& operator=(const HostTensor&) = delete;

    char* m_allocated_buffer_pool;
    char* m_aligned_buffer_pool;
    size_t m_buffer_size;
};
