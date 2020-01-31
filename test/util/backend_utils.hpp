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

#ifdef NGRAPH_UNIT_TEST_OPENVINO_ENABLE
#include <ie_core.hpp>
#include <string>
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

Blob::Ptr fill_blob(SizeVector shape, std::vector<float> data);

class Handle;

namespace ov_runtime
{
    class Backend;
    class Tensor
    {
    public:
        std::vector<float> data;
        PartialShape shape;
        ngraph::element::Type type;

        Shape get_shape() { return shape.to_shape(); }
        PartialShape get_partial_shape() { return shape; }
        explicit Tensor(ngraph::element::Type type, PartialShape ps)
            : type(type)
            , shape(ps)
        {
        }
        explicit Tensor(ngraph::element::Type type, Shape ps)
            : type(type)
            , shape(ps)
        {
        }

        const element::Type& get_element_type() const { return type; }
        size_t get_element_count() { return shape_size(get_shape()); }
        void set_stale(bool flag) {}
        void copy_from(ov_runtime::Tensor t)
        {
            data = t.data;
            shape = t.shape;
            type = t.type;
        }
    };
}

class Executable
{
private:
    CNNNetwork network;
    std::string device;

public:
    Executable(std::shared_ptr<Function> func, std::string _device)
    {
        network = CNNNetwork(func);
        device = _device;
    }

    bool call_with_validate(const vector<shared_ptr<ov_runtime::Tensor>>& outputs,
                            const vector<shared_ptr<ov_runtime::Tensor>>& inputs)
    {
        Core ie;

        //  Loading model to the plugin (BACKEND_NAME)
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, device);
        //  Create infer request
        InferRequest inferRequest = exeNetwork.CreateInferRequest();
        //  Prepare input and output blobs
        InputsDataMap inputInfo = network.getInputsInfo();

        if (inputInfo.size() != inputs.size())
        {
            THROW_IE_EXCEPTION << "Function inputs number differ from number of given inputs";
        }

        size_t i = 0;
        for (auto& it : inputInfo)
        {
            inferRequest.SetBlob(
                it.first, fill_blob(it.second->getTensorDesc().getDims(), inputs[i++]->data));
        }

        //  Prepare output blobs
        std::string output_name = network.getOutputsInfo().begin()->first;

        inferRequest.Infer();
        Blob::Ptr output = inferRequest.GetBlob(output_name);

        float* output_ptr = output->buffer().as<float*>();
        // TODO: how to get size without explicit calculation?
        size_t size = 1;
        for (const auto& dim : output->getTensorDesc().getDims())
        {
            size *= dim;
        }
        //  Vector initialization from pointer
        std::vector<float> result(output_ptr, output_ptr + size);
        outputs[0]->data = result;
        return true;
    }
};

template <class T>
void copy_data(std::shared_ptr<ov_runtime::Tensor> t, const std::vector<T>& data);

class ov_runtime::Backend
{
private:
    string device;

public:
    static std::shared_ptr<ov_runtime::Backend> create(std::string device,
                                                       bool must_support_dynamic = false)
    {
        return std::shared_ptr<Backend>(new Backend(device));
    }

    Backend(std::string _device)
        : device(_device)
    {
    }

    std::shared_ptr<ov_runtime::Tensor> create_tensor(ngraph::element::Type type,
                                                      ngraph::Shape shape)
    {
        return std::shared_ptr<ov_runtime::Tensor>(new ov_runtime::Tensor(type, shape));
    }

    template <typename T>
    std::shared_ptr<ov_runtime::Tensor>
        create_tensor(ngraph::element::Type type, ngraph::Shape shape, T* data)
    {
        auto tensor = std::shared_ptr<ov_runtime::Tensor>(new ov_runtime::Tensor(type, shape));
        size_t size = 1;
        for (auto x : shape)
        {
            size *= x;
        }
        vector<T> v(data, data + size);
        copy_data(tensor, v);
        return tensor;
    }

    template <class T>
    std::shared_ptr<ov_runtime::Tensor> create_tensor(ngraph::Shape shape)
    {
        return std::shared_ptr<ov_runtime::Tensor>(
            new ov_runtime::Tensor(ngraph::element::from<T>(), shape));
    }

    std::shared_ptr<ov_runtime::Tensor> create_dynamic_tensor(ngraph::element::Type type,
                                                              ngraph::PartialShape shape)
    {
        return std::shared_ptr<ov_runtime::Tensor>(new ov_runtime::Tensor(type, shape));
    }

    bool supports_dynamic_tensors() { return true; }
    std::shared_ptr<Executable> compile(std::shared_ptr<Function> func)
    {
        return std::shared_ptr<Executable>(new Executable(func, device));
    }
};

template <class T>
std::vector<T> read_vector(std::shared_ptr<ov_runtime::Tensor> tv)
{
    std::vector<T> v(tv->data.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = tv->data[i];
    }
    return v;
}

template <class T>
void copy_data(std::shared_ptr<ov_runtime::Tensor> t, const std::vector<T>& data)
{
    t->data.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        t->data[i] = data[i];
    }
}

#endif