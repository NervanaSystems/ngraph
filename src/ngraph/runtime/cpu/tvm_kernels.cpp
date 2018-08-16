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
#include <topi/transform.h>
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"

#include "tvm_kernels.hpp"

using namespace ngraph::runtime::cpu;
TVMInstance::TVMInstance()
{
    NGRAPH_INFO << "Creating TVMInstance";
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

tvm_func reshape(const std::unique_ptr<TVMInstance>& tvm_instance,
                 const ngraph::Node* node,
                 const std::vector<TensorViewWrapper>& args,
                 const std::vector<TensorViewWrapper>& out,
                 std::unordered_map<std::string, void*>& tensor_data)
{
    auto reshape = static_cast<const ngraph::op::Reshape*>(node);
    auto input_order = reshape->get_input_order();

    tvm::Array<tvm::Expr> ae;
    for (auto& v : args[0].get_shape())
        ae.push_back(tvm::make_const(tvm::Int(32), v));
    auto A = tvm::placeholder(ae, tvm::Float(32), "a");

    tvm::Array<tvm::Expr> newshape;
    tvm::Tensor R;
    if (reshape->get_is_transpose())
    {
        // topi axes
        for (auto& v : input_order)
            newshape.push_back(tvm::make_const(tvm::Int(32), v));
        R = topi::transpose(A, newshape);
    }
    else
    {
        // output shape
        for (auto& v : out[0].get_shape())
            newshape.push_back(tvm::make_const(tvm::Int(32), v));
        R = topi::reshape(A, newshape);
    }

    return tvm_unary_func({A, R}, tvm_instance, node, args, out, tensor_data);
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

    // use default schedule
    auto schedule = topi::x86::default_schedule(tvm_instance->target(), {R});

    auto lowered = tvm::lower(schedule, {I, W, R}, "func", binds, tvm_instance->config());
    auto func = tvm_instance->get_func(lowered);

    // get tensor_data ptrs
    auto& it = tensor_data[args[0].get_name()];
    auto& wt = tensor_data[args[1].get_name()];
    auto& rt = tensor_data[out[0].get_name()];
    return [&, func, args, out](CPURuntimeContext* ctx) {
        std::vector<int64_t> i_shape(args[0].get_shape().begin(), args[0].get_shape().end());
        std::vector<int64_t> w_shape(args[1].get_shape().begin(), args[1].get_shape().end());
        std::vector<int64_t> r_shape(out[0].get_shape().begin(), out[0].get_shape().end());
        DLTensor i = tvm_instance->create_dltensor(DLType_Float32, i_shape.size(), &i_shape[0], it);
        DLTensor w = tvm_instance->create_dltensor(DLType_Float32, w_shape.size(), &w_shape[0], wt);
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
                 {TI(ngraph::op::Reshape), reshape},
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
    NGRAPH_INFO << "TVM kernel registered for ngraph op: " << node->get_friendly_name();
    auto& functors = external_function->get_functors();
    auto& tvm_instance = external_function->get_tvm_instance();
    auto& tensor_data = external_function->get_tensor_data();
    auto func = tvm_funcs[key](tvm_instance, node, args, out, tensor_data);
    functors.emplace_back(func);
    return true;
}
