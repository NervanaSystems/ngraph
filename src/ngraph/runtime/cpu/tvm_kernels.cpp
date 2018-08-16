/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

#include <dmlc/logging.h>
#include <ngraph/util.hpp>
#include <topi/nn.h>
#include <topi/nn/batch_norm.h>
#include <topi/nn/pooling.h>
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/relu.hpp"

#include "tvm_kernels.hpp"

using namespace ngraph::runtime::cpu;
TVMInstance::TVMInstance()
{
    NGRAPH_DEBUG << "Creating TVMInstance";
    m_config = tvm::build_config();
    m_target = tvm::target::llvm();
    m_dl_ctx.device_type = static_cast<DLDeviceType>(kDLCPU);
    m_dl_ctx.device_id = 0;
}
TVMInstance::~TVMInstance()
{
}

DLTensor TVMInstance::create_dltensor(const DLDataType& type,
                                      const size_t ndim,
                                      tvm_index_t* shape,
                                      void* data)
{
    DLTensor t;
    t.ctx = m_dl_ctx;
    t.ndim = ndim;
    t.dtype = type;
    t.shape = static_cast<int64_t*>(shape);
    t.strides = nullptr;
    t.byte_offset = 0;
    t.data = data;
    return t;
}

static const DLDataType DLType_Float32{kDLFloat, 32, 1};

template <>
tvm::PackedFunc
    tvm_kernel::unary_elemwise_builder<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                              const UnaryElemwiseFunc& topi_func)
{
    tvm::Var n("n");
    auto A = tvm::placeholder({n}, tvm::Float(32), "a");

    auto R = topi_func(A, "tensor", topi::kElementWise);

    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;

    auto schedule = topi::x86::default_schedule(tvm_instance->target(), {R});
    auto lowered = tvm::lower(schedule, {A, R}, "func", binds, tvm_instance->config());
    auto module =
        tvm::build(lowered, tvm_instance->target(), tvm::Target(), tvm_instance->config());
    // store module to keep its lifetime
    tvm_instance->add_module(module);
    return module->GetFunction("func", false);
}

template <>
void tvm_kernel::unary_elemwise_kernel<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                              const tvm::PackedFunc& func,
                                              void* input,
                                              void* output,
                                              size_t count)
{
    std::cout << "running tvm_kernel::unary_elemwise_kernel" << std::endl;
    int64_t dlshape[] = {static_cast<int64_t>(count)};
    DLTensor a = tvm_instance->create_dltensor(DLType_Float32, 1, dlshape, input);
    DLTensor r = tvm_instance->create_dltensor(DLType_Float32, 1, dlshape, output);

    func(&a, &r);
}

template <>
tvm::PackedFunc
    tvm_kernel::binary_elemwise_builder<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                               const BinaryElemwiseFunc& topi_func)
{
    tvm::Var n("n");
    auto A = tvm::placeholder({n}, tvm::Float(32), "a");
    auto B = tvm::placeholder({n}, tvm::Float(32), "b");

    auto R = topi_func(A, B, "tensor", topi::kBroadcast);

    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;

    auto schedule = topi::x86::default_schedule(tvm_instance->target(), {R});
    auto lowered = tvm::lower(schedule, {A, B, R}, "func", binds, tvm_instance->config());
    auto module =
        tvm::build(lowered, tvm_instance->target(), tvm::Target(), tvm_instance->config());
    // store module to keep its lifetime
    tvm_instance->add_module(module);
    return module->GetFunction("func", false);
}

template <>
void tvm_kernel::binary_elemwise_kernel<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                               const tvm::PackedFunc& func,
                                               void* input0,
                                               void* input1,
                                               void* output,
                                               size_t count)
{
    std::cout << "running tvm_kernel::binary_elemwise_kernel" << std::endl;
    int64_t dlshape[] = {static_cast<int64_t>(count)};
    DLTensor a = tvm_instance->create_dltensor(DLType_Float32, 1, dlshape, input0);
    DLTensor b = tvm_instance->create_dltensor(DLType_Float32, 1, dlshape, input1);
    DLTensor r = tvm_instance->create_dltensor(DLType_Float32, 1, dlshape, output);

    func(&a, &b, &r);
}

template <>
tvm::PackedFunc
    tvm_kernel::transpose_builder<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                         const size_t in_rank,
                                         const std::vector<size_t>& in_shape,
                                         const size_t out_rank,
                                         const std::vector<size_t>& axes)
{
    std::cout << in_rank << " " << out_rank << " axes: " << axes.size() << std::endl;

    tvm::Array<tvm::Expr> in_dlshape;
    for (size_t i = 0; i < in_rank; ++i)
    {
        tvm::Var n("n_" + std::to_string(i));
        in_dlshape.push_back(n);
    }
    std::cout << ngraph::vector_to_string(axes) << std::endl;
    tvm::Array<tvm::Expr> out_axes;
    for (size_t i = 0; i < out_rank; ++i)
    {
        std::cout << "axes[i]: " << axes[i] << std::endl;
        out_axes.push_back(axes[i]);
    }
    auto A = tvm::placeholder(in_dlshape, tvm::Float(32), "a");

    auto R = topi::transpose(A, out_axes);

    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;

    auto schedule = topi::x86::default_schedule(tvm_instance->target(), {R});
    auto lowered = tvm::lower(schedule, {A, R}, "func", binds, tvm_instance->config());
    auto module =
        tvm::build(lowered, tvm_instance->target(), tvm::Target(), tvm_instance->config());
    // store module to keep its lifetime
    tvm_instance->add_module(module);
    return module->GetFunction("func", false);
}

template <>
void tvm_kernel::transpose_kernel<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                         const tvm::PackedFunc& func,
                                         void* input,
                                         void* output,
                                         Shape input_shape,
                                         Shape output_shape)
{
    std::cout << "running tvm_kernel::transpose_kernel" << std::endl;

    std::cout << ngraph::vector_to_string(input_shape) << std::endl;
    std::cout << ngraph::vector_to_string(output_shape) << std::endl;
    //    int64_t* in_dlshape = reinterpret_cast<int64_t*>(&input_shape[0]);
    //    int64_t* out_dlshape = reinterpret_cast<int64_t*>(&output_shape[0]);

    std::vector<int64_t> in_dlshape(input_shape.size());
    for (int i = 0; i < input_shape.size(); ++i)
    {
        in_dlshape[i] = (int64_t)input_shape[i];
    }
    std::vector<int64_t> out_dlshape(output_shape.size());
    for (int i = 0; i < output_shape.size(); ++i)
    {
        out_dlshape[i] = (int64_t)output_shape[i];
    }
    std::cout << ngraph::vector_to_string(in_dlshape) << std::endl;
    std::cout << ngraph::vector_to_string(out_dlshape) << std::endl;

    DLTensor a =
        tvm_instance->create_dltensor(DLType_Float32, in_dlshape.size(), &in_dlshape[0], input);
    DLTensor r =
        tvm_instance->create_dltensor(DLType_Float32, out_dlshape.size(), &out_dlshape[0], output);

    func(&a, &r);
    std::cout << "tvm reshape output: " << std::endl;
    for (int i = 0; i < ngraph::shape_size(out_dlshape); ++i)
    {
        std::cout << static_cast<float*>(r.data)[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < ngraph::shape_size(out_dlshape); ++i)
    {
        std::cout << static_cast<float*>(output)[i] << " ";
    }
    std::cout << std::endl;
}

#define TI(x) std::type_index(typeid(x))
using tvm_func = std::function<void(CPURuntimeContext* ctx)>;

#define TVM_BINARY_FUNC(OP)                                                                        \
    [](const std::unique_ptr<TVMInstance>& tvm_instance,                                           \
       const ngraph::Node* node,                                                                   \
       const std::vector<TensorViewWrapper>& args,                                                 \
       const std::vector<TensorViewWrapper>& out,                                                  \
       std::unordered_map<std::string, void*>& tensor_data) {                                      \
        tvm::Array<tvm::Expr> ae, be;                                                              \
        for (auto& v : args[0].get_shape())                                                        \
            ae.push_back(tvm::make_const(tvm::Int(32), v));                                        \
        for (auto& v : args[1].get_shape())                                                        \
            be.push_back(tvm::make_const(tvm::Int(32), v));                                        \
        auto A = tvm::placeholder(ae, tvm::Float(32), "a");                                        \
        auto B = tvm::placeholder(be, tvm::Float(32), "b");                                        \
        auto R = OP(A, B, "tensor", topi::kBroadcast);                                             \
        return tvm_binary_func({A, B, R}, tvm_instance, node, args, out, tensor_data);             \
    }
tvm_func tvm_binary_func(const tvm::Array<tvm::Tensor>& G,
                         const std::unique_ptr<TVMInstance>& tvm_instance,
                         const ngraph::Node* node,
                         const std::vector<TensorViewWrapper>& args,
                         const std::vector<TensorViewWrapper>& out,
                         std::unordered_map<std::string, void*>& tensor_data)
{
    // create tvm func
    auto func = tvm_instance->get_func(G);

    // get tensor_data ptrs
    auto& at = tensor_data[args[0].get_name()];
    auto& bt = tensor_data[args[1].get_name()];
    auto& rt = tensor_data[out[0].get_name()];
    return [&, func, args, out](CPURuntimeContext* ctx) {
        std::vector<int64_t> a_shape(args[0].get_shape().begin(), args[0].get_shape().end());
        std::vector<int64_t> b_shape(args[1].get_shape().begin(), args[1].get_shape().end());
        std::vector<int64_t> r_shape(out[0].get_shape().begin(), out[0].get_shape().end());
        DLTensor a = tvm_instance->create_dltensor(DLType_Float32, a_shape.size(), &a_shape[0], at);
        DLTensor b = tvm_instance->create_dltensor(DLType_Float32, b_shape.size(), &b_shape[0], bt);
        DLTensor r = tvm_instance->create_dltensor(DLType_Float32, r_shape.size(), &r_shape[0], rt);
        func(&a, &b, &r);
    };
}
tvm_func tvm_unary_func(const tvm::Array<tvm::Tensor>& G,
                        const std::unique_ptr<TVMInstance>& tvm_instance,
                        const ngraph::Node* node,
                        const std::vector<TensorViewWrapper>& args,
                        const std::vector<TensorViewWrapper>& out,
                        std::unordered_map<std::string, void*>& tensor_data)
{
    // create tvm func
    auto func = tvm_instance->get_func(G);

    // get tensor_data ptrs
    auto& at = tensor_data[args[0].get_name()];
    auto& rt = tensor_data[out[0].get_name()];
    return [&, func, args, out](CPURuntimeContext* ctx) {
        std::vector<int64_t> a_shape(args[0].get_shape().begin(), args[0].get_shape().end());
        std::vector<int64_t> r_shape(out[0].get_shape().begin(), out[0].get_shape().end());
        DLTensor a = tvm_instance->create_dltensor(DLType_Float32, a_shape.size(), &a_shape[0], at);
        DLTensor r = tvm_instance->create_dltensor(DLType_Float32, r_shape.size(), &r_shape[0], rt);
        func(&a, &r);
    };
}
tvm_func batch_norm(const std::unique_ptr<TVMInstance>& tvm_instance,
                    const ngraph::Node* node,
                    const std::vector<TensorViewWrapper>& args,
                    const std::vector<TensorViewWrapper>& out,
                    std::unordered_map<std::string, void*>& tensor_data)
{
    const ngraph::op::BatchNorm* batchnorm = static_cast<const ngraph::op::BatchNorm*>(node);
    // create tvm module
    tvm::Var n, c, h, w;
    auto x = tvm::placeholder({n, c, h, w}, tvm::Float(32), "x");
    auto gamma = tvm::placeholder({c}, tvm::Float(32), "gamma");
    auto beta = tvm::placeholder({c}, tvm::Float(32), "beta");
    auto mean = tvm::placeholder({c}, tvm::Float(32), "mean");
    auto var = tvm::placeholder({c}, tvm::Float(32), "var");
    auto eps = batchnorm->get_eps_value();

    auto R = topi::nn::batch_norm_inference(x, gamma, beta, mean, var, eps, false);
    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;

    auto schedule = topi::x86::default_schedule(tvm_instance->target(), {R});
    auto lowered =
        tvm::lower(schedule, {x, gamma, beta, mean, var, R}, "func", binds, tvm_instance->config());
    auto func = tvm_instance->get_func(lowered);

    // get tensor_data ptrs
    auto& xt = tensor_data[args[2].get_name()];
    auto& gammat = tensor_data[args[0].get_name()];
    auto& betat = tensor_data[args[1].get_name()];
    auto& meant = tensor_data[args[3].get_name()];
    auto& vart = tensor_data[args[4].get_name()];
    auto& rt = tensor_data[out[0].get_name()];

    return [&, func, args, out](CPURuntimeContext* ctx) {
        std::vector<int64_t> x_shape(args[2].get_shape().begin(), args[2].get_shape().end());
        std::vector<int64_t> gamma_shape(args[0].get_shape().begin(), args[0].get_shape().end());
        std::vector<int64_t> beta_shape(args[1].get_shape().begin(), args[1].get_shape().end());
        std::vector<int64_t> mean_shape(args[3].get_shape().begin(), args[3].get_shape().end());
        std::vector<int64_t> var_shape(args[4].get_shape().begin(), args[4].get_shape().end());
        std::vector<int64_t> r_shape(out[0].get_shape().begin(), out[0].get_shape().end());

        DLTensor x = tvm_instance->create_dltensor(DLType_Float32, x_shape.size(), &x_shape[0], xt);
        DLTensor gamma = tvm_instance->create_dltensor(
            DLType_Float32, gamma_shape.size(), &gamma_shape[0], gammat);
        DLTensor beta =
            tvm_instance->create_dltensor(DLType_Float32, beta_shape.size(), &beta_shape[0], betat);
        DLTensor mean =
            tvm_instance->create_dltensor(DLType_Float32, mean_shape.size(), &mean_shape[0], meant);
        DLTensor var =
            tvm_instance->create_dltensor(DLType_Float32, var_shape.size(), &var_shape[0], vart);
        DLTensor r = tvm_instance->create_dltensor(DLType_Float32, r_shape.size(), &r_shape[0], rt);

        func(&x, &gamma, &beta, &mean, &var, &r);
    };
}

tvm_func convolution(const std::unique_ptr<TVMInstance>& tvm_instance,
                     const ngraph::Node* node,
                     const std::vector<TensorViewWrapper>& args,
                     const std::vector<TensorViewWrapper>& out,
                     std::unordered_map<std::string, void*>& tensor_data)
{
    auto convolution = static_cast<const ngraph::op::Convolution*>(node);
    // create tvm module
    tvm::Array<tvm::Expr> ae, be;
    for (auto& v : args[0].get_shape())
        ae.push_back(tvm::make_const(tvm::Int(32), v));
    for (auto& v : args[1].get_shape())
        be.push_back(tvm::make_const(tvm::Int(32), v));
    auto I = tvm::placeholder(ae, tvm::Float(32), "I");
    auto W = tvm::placeholder(be, tvm::Float(32), "W");
    auto s = convolution->get_window_movement_strides();
    auto p = convolution->get_padding_above();
    auto R = topi::conv2d_nchw(I, W, int(p[0]), int(p[1]), int(s[0]), int(s[1]));

    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
    auto schedule = topi::x86::default_schedule(tvm_instance->target(), {R});
    auto lowered = tvm::lower(schedule, {I, W, R}, "func", binds, tvm_instance->config());
    auto func = tvm_instance->get_func(lowered);

    // get tensor_data ptrs
    auto& it = tensor_data[args[0].get_name()];
    auto& wt = tensor_data[args[1].get_name()];
    auto& rt = tensor_data[out[0].get_name()];
    return [&, func, args, out](CPURuntimeContext* ctx) {
        std::vector<int64_t> i_shape(args[0].get_shape().begin(), args[0].get_shape().end());
        DLTensor i = tvm_instance->create_dltensor(DLType_Float32, i_shape.size(), &i_shape[0], it);
        std::vector<int64_t> w_shape(args[1].get_shape().begin(), args[1].get_shape().end());
        DLTensor w = tvm_instance->create_dltensor(DLType_Float32, w_shape.size(), &w_shape[0], wt);
        std::vector<int64_t> r_shape(out[0].get_shape().begin(), out[0].get_shape().end());
        DLTensor r = tvm_instance->create_dltensor(DLType_Float32, r_shape.size(), &r_shape[0], rt);
        func(&i, &w, &r);
    };
}
tvm_func pool_max(const std::unique_ptr<TVMInstance>& tvm_instance,
                  const ngraph::Node* node,
                  const std::vector<TensorViewWrapper>& args,
                  const std::vector<TensorViewWrapper>& out,
                  std::unordered_map<std::string, void*>& tensor_data)
{
    auto pool = static_cast<const ngraph::op::MaxPool*>(node);
    tvm::Array<tvm::Expr> ae;
    for (auto& v : args[0].get_shape())
        ae.push_back(tvm::make_const(tvm::Int(32), v));
    auto I = tvm::placeholder(ae, tvm::Float(32), "i");

    auto k = pool->get_window_shape();
    auto s = pool->get_window_movement_strides();
    auto pb = pool->get_padding_below();
    auto pa = pool->get_padding_above();
    tvm::Array<tvm::Expr> ke, se, pe;
    for (auto& v : k)
        ke.push_back(tvm::make_const(tvm::Int(32), v));
    for (auto& v : s)
        se.push_back(tvm::make_const(tvm::Int(32), v));
    for (auto& v : pb)
        pe.push_back(tvm::make_const(tvm::Int(32), v));
    for (auto& v : pa)
        pe.push_back(tvm::make_const(tvm::Int(32), v));
    auto R = topi::nn::pool(I, ke, se, pe, topi::nn::kMaxPool, false);
    return tvm_unary_func({I, R}, tvm_instance, node, args, out, tensor_data);
}
tvm_func pool_avg(const std::unique_ptr<TVMInstance>& tvm_instance,
                  const ngraph::Node* node,
                  const std::vector<TensorViewWrapper>& args,
                  const std::vector<TensorViewWrapper>& out,
                  std::unordered_map<std::string, void*>& tensor_data)
{
    auto pool = static_cast<const ngraph::op::AvgPool*>(node);
    tvm::Array<tvm::Expr> ae;
    for (auto& v : args[0].get_shape())
        ae.push_back(tvm::make_const(tvm::Int(32), v));
    auto I = tvm::placeholder(ae, tvm::Float(32), "i");

    auto k = pool->get_window_shape();
    auto s = pool->get_window_movement_strides();
    auto pb = pool->get_padding_below();
    auto pa = pool->get_padding_above();
    tvm::Array<tvm::Expr> ke, se, pe;
    for (auto& v : k)
        ke.push_back(tvm::make_const(tvm::Int(32), v));
    for (auto& v : s)
        se.push_back(tvm::make_const(tvm::Int(32), v));
    for (auto& v : pb)
        pe.push_back(tvm::make_const(tvm::Int(32), v));
    for (auto& v : pa)
        pe.push_back(tvm::make_const(tvm::Int(32), v));
    auto count_include_pad = pool->get_include_padding_in_avg_computation();
    auto R = topi::nn::pool(I, ke, se, pe, topi::nn::kAvgPool, false, "NCHW", count_include_pad);
    return tvm_unary_func({I, R}, tvm_instance, node, args, out, tensor_data);
}
tvm_func relu(const std::unique_ptr<TVMInstance>& tvm_instance,
              const ngraph::Node* node,
              const std::vector<TensorViewWrapper>& args,
              const std::vector<TensorViewWrapper>& out,
              std::unordered_map<std::string, void*>& tensor_data)
{
    tvm::Array<tvm::Expr> ae;
    for (auto& v : args[0].get_shape())
        ae.push_back(tvm::make_const(tvm::Int(32), v));
    auto A = tvm::placeholder(ae, tvm::Float(32), "a");
    auto R = topi::relu<float>(A);
    return tvm_unary_func({A, R}, tvm_instance, node, args, out, tensor_data);
}
tvm_func matmul(const std::unique_ptr<TVMInstance>& tvm_instance,
                const ngraph::Node* node,
                const std::vector<TensorViewWrapper>& args,
                const std::vector<TensorViewWrapper>& out,
                std::unordered_map<std::string, void*>& tensor_data)
{
    tvm::Array<tvm::Expr> ae, be;
    for (auto& v : args[0].get_shape())
        ae.push_back(tvm::make_const(tvm::Int(32), v));
    for (auto& v : args[1].get_shape())
        be.push_back(tvm::make_const(tvm::Int(32), v));
    auto A = tvm::placeholder(ae, tvm::Float(32), "a");
    auto B = tvm::placeholder(be, tvm::Float(32), "b");
    auto R = topi::matmul(A, B, false, false, "tensor", topi::kMatMul);
    return tvm_binary_func({A, B, R}, tvm_instance, node, args, out, tensor_data);
}
std::unordered_map<std::type_index,
                   std::function<tvm_func(const std::unique_ptr<TVMInstance>&,
                                          const ngraph::Node*,
                                          const std::vector<TensorViewWrapper>&,
                                          const std::vector<TensorViewWrapper>&,
                                          std::unordered_map<std::string, void*>&)>>
    tvm_funcs = {{TI(ngraph::op::Divide), TVM_BINARY_FUNC(topi::divide)},
                 {TI(ngraph::op::Add), TVM_BINARY_FUNC(topi::add)},
                 {TI(ngraph::op::BatchNorm), batch_norm},
                 {TI(ngraph::op::Convolution), convolution},
                 {TI(ngraph::op::Dot), matmul},
                 {TI(ngraph::op::AvgPool), pool_avg},
                 {TI(ngraph::op::MaxPool), pool_max},
                 {TI(ngraph::op::Relu), relu}};

bool ngraph::runtime::cpu::build_tvm_functor(CPU_ExternalFunction* external_function,
                                             const ngraph::Node* node,
                                             const std::vector<TensorViewWrapper>& args,
                                             const std::vector<TensorViewWrapper>& out)
{
    auto key = TI(*node);
    if (tvm_funcs.find(key) == tvm_funcs.end())
    {
        return false;
    }
    NGRAPH_DEBUG << "TVM kernel registered for ngraph op: " << node->get_friendly_name();
    auto& functors = external_function->get_functors();
    auto& tvm_instance = external_function->get_tvm_instance();
    auto& tensor_data = external_function->get_tensor_data();
    auto func = tvm_funcs[key](tvm_instance, node, args, out, tensor_data);
    functors.emplace_back(func);
    return true;
}
