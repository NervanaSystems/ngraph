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

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace dynamic_wrapper
        {
            class DynamicWrapperBackend;
            //class DynamicWrapperExecutable;
            class DynamicTensor;
        }
    }
}

class ngraph::runtime::dynamic_wrapper::DynamicWrapperBackend : public Backend
{
public:
    DynamicWrapperBackend(std::unique_ptr<ngraph::runtime::Backend> wrapped_backend);

    std::shared_ptr<Tensor>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<Tensor> create_tensor(const element::Type& type, const Shape& shape) override;

    std::shared_ptr<Tensor> create_dynamic_tensor(const element::Type& type,
                                                  const PartialShape& shape) override;

    bool supports_dynamic_tensors() override { return true; }
    std::shared_ptr<Executable> compile(std::shared_ptr<Function> function,
                                        bool enable_performance_data = false) override;

private:
    std::shared_ptr<ngraph::runtime::Backend> m_wrapped_backend;
};

#if 0
class ngraph::runtime::dynamic_wrapper::DynamicWrapperExecutable : public Executable
{
public:
    DynamicWrapperExecutable(std::shared_ptr<Function> function, bool enable_performance_collection = false);
    bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;
};
#endif

class ngraph::runtime::dynamic_wrapper::DynamicTensor : public ngraph::runtime::Tensor
{
public:
    DynamicTensor(const ngraph::element::Type& element_type,
                  const PartialShape& shape,
                  const runtime::Backend* parent,
                  const std::shared_ptr<ngraph::runtime::Backend>& wrapped_backend);
    virtual const ngraph::Shape& get_shape() const override;
    virtual void write(const void* p, size_t offset, size_t n) override;
    virtual void read(void* p, size_t offset, size_t n) const override;
    virtual void copy_from(const ngraph::runtime::Tensor& source) override;
    void set_type_and_shape(const element::Type& et, const Shape& shape);
    void clear_type_and_shape();

private:
    std::shared_ptr<ngraph::runtime::Tensor> m_wrapped_tensor;
    std::shared_ptr<ngraph::runtime::Backend> m_wrapped_backend;
};
