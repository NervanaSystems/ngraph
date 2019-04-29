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
#include "compiler.hpp"
#include <memory>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/LLVMIR/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/runtime/cpu/mlir/dialect/dialect.hpp"
#include "ngraph/runtime/cpu/mlir/dialect/ops.hpp"
#include "ngraph/runtime/cpu/mlir/dialect/type.hpp"
#include "ngraph/runtime/cpu/mlir/lowerer.hpp"
#include "ngraph/type/element_type.hpp"

using llvm::SmallVector;
using llvm::make_unique;
using namespace ngraph::runtime::cpu;

#define COMPILE_OP_DECL(op_name)                                                                   \
    create_op<op_name>(MLIRCompiler & compiler, const ngraph::Node* ng_node)

namespace ngraph
{
    void MLIRCompiler::init_mlir() 
    { 
        mlir::registerDialect<NGDialect>(); 
        // Register any LLVM command line options
        llvm::cl::ParseEnvironmentOptions("ngraph", "MLIR_LLVM_OPTIONS", "");
    }
    void MLIRCompiler::compile()
    {
        build_module(); // MLIR gen
        lower_dialect();
        lower_to_llvm();
        // JIT and invoke main
    }

    void MLIRCompiler::build_module()
    {
        // initialize an empty module
        m_module = make_unique<mlir::Module>(&m_context);

        TypeList args_type_list, result_type_list;
        build_tensors_list();
        NGRAPH_ASSERT(m_ip_tensors.size() != 0) << "Cannot have empty inputs list";
        NGRAPH_ASSERT(m_op_tensors.size() != 0) << "Cannot have empty outputs list";

        for (auto tensor : m_ip_tensors)
        {
            args_type_list.push_back(get_mlir_type(tensor));
        }

        for (auto tensor : m_op_tensors)
        {
            result_type_list.push_back(get_mlir_type(tensor));
        }

        auto func_type = mlir::FunctionType::get(args_type_list, result_type_list, &m_context);
        auto function =
            make_unique<mlir::Function>(mlir::UnknownLoc::get(&m_context), "main", func_type);
        function->addEntryBlock();

        // populate Tensor->Value maps
        int i = 0;
        for (auto tensor : m_ip_tensors)
        {
            mlir::Value* arg = function->getArgument(i);
            TensorInfo tensor_info{arg};
            m_tensor_to_value_map.insert(TensorToInfo(tensor, tensor_info));
            i++;
        }

        // create builder
        m_builder = llvm::make_unique<mlir::FuncBuilder>(function.get());
        build_ng_dialect();
        m_module->getFunctions().push_back(function.release());
        if (std::getenv("NGRAPH_MLIR_DUMP_ALL") != nullptr)
        {
            m_module->dump();
        }
    }

    /// Collects input and output tensors to this sub-graph
    void MLIRCompiler::build_tensors_list()
    {
        for (const auto node : m_sub_graph)
        {
            // get all nodes output tensors
            // if an output has a use out of the subgraph, it is an output tensor, else a temp.
            for (auto i = 0; i < node->get_output_size(); i++)
            {
                const std::set<descriptor::Input*>& inputs = node->get_output_inputs(i);
                auto tensor = node->get_output_tensor_ptr(i);
                for (auto ip : inputs)
                {
                    bool out_of_subgraph =
                        (std::find(std::begin(m_sub_graph),
                                   std::end(m_sub_graph),
                                   ip->get_node().get()) == std::end(m_sub_graph));
                    if (out_of_subgraph)
                    {
                        // we found a use out of subgraph, consider this an output tensor
                        // those would be added as return value for the mlir func
                        if (std::find(std::begin(m_op_tensors),
                                      std::end(m_op_tensors),
                                      tensor.get()) == std::end(m_op_tensors))
                        {
                            m_op_tensors.push_back(tensor.get());
                        }
                    }
                }
            }

            // get over all input tensors
            for (const auto arg : node->get_arguments())
            {
                bool out_of_subgraph =
                    (std::find(std::begin(m_sub_graph), std::end(m_sub_graph), arg.get()) ==
                     std::end(m_sub_graph));
                if (out_of_subgraph)
                {
                    for (auto i = 0; i < arg->get_output_size(); i++)
                    {
                        auto tensor = arg->get_output_tensor_ptr(i);
                        if (std::find(std::begin(m_ip_tensors),
                                      std::end(m_ip_tensors),
                                      tensor.get()) == std::end(m_ip_tensors))
                        {
                            m_ip_tensors.push_back(tensor.get());
                        }
                    }
                }
            }
        }
    }

    mlir::Type MLIRCompiler::get_mlir_type(const descriptor::Tensor* tensor)
    {
        SmallVector<int64_t, 4> shape;
        for (auto d : tensor->get_shape())
        {
            shape.push_back(d);
        }

        return NGTensorType::get(&m_context, get_mlir_type(tensor->get_element_type()), shape);
    }

    mlir::Type MLIRCompiler::get_mlir_type(const element::Type& type)
    {
        switch (type.get_type_enum())
        {
        case ngraph::element::Type_t::undefined:
        case ngraph::element::Type_t::dynamic:
        case ngraph::element::Type_t::boolean:
        case ngraph::element::Type_t::bf16:
        default: NGRAPH_ASSERT(false) << "MLIR: Unsupported NGraph types"; break;
        case ngraph::element::Type_t::f32: return mlir::FloatType::getF32(&m_context);
        case ngraph::element::Type_t::f64: return mlir::FloatType::getF64(&m_context);
        case ngraph::element::Type_t::i8:
        case ngraph::element::Type_t::u8: return mlir::IntegerType::get(8, &m_context);
        case ngraph::element::Type_t::i16:
        case ngraph::element::Type_t::u16: return mlir::IntegerType::get(16, &m_context);
        case ngraph::element::Type_t::i32:
        case ngraph::element::Type_t::u32: return mlir::IntegerType::get(32, &m_context);
        case ngraph::element::Type_t::i64:
        case ngraph::element::Type_t::u64: return mlir::IntegerType::get(64, &m_context);
        }
        NGRAPH_ASSERT(false) << "Unreachable";
        return mlir::Type();
    }

    void MLIRCompiler::update_tensor_value(descriptor::Tensor* tensor, mlir::Value* value)
    {
        NGRAPH_ASSERT(m_tensor_to_value_map.find(tensor) == m_tensor_to_value_map.end())
            << "tensor value already defined";
        TensorInfo tensor_info{value};
        m_tensor_to_value_map.insert(TensorToInfo(tensor, tensor_info));
    }

    MLIRCompiler::TensorInfo MLIRCompiler::get_tensor_value(descriptor::Tensor* tensor)
    {
        auto it = m_tensor_to_value_map.find(tensor);

        NGRAPH_ASSERT(it != m_tensor_to_value_map.end()) << "Undefined tensor";

        return it->second;
    }

    void MLIRCompiler::lower_dialect()
    {
        mlir::PassManager pm;
        pm.addPass(createDialectLoweringPass());
        pm.addPass(mlir::createCanonicalizerPass());

        pm.run(m_module.get());
        if (std::getenv("NGRAPH_MLIR_DUMP_ALL") != nullptr)
        {
            m_module->dump();
        }
    }

    void MLIRCompiler::lower_to_llvm()
    {
        mlir::PassManager pm;
        // Lower affine ops
        pm.addPass(mlir::createLowerAffinePass());
        auto rr = pm.run(m_module.get());
        (void)rr;
        assert(succeeded(rr) && "affine loop lowering failed");
        // std to llvm dialect
        auto converter = mlir::createStdToLLVMConverter();
        auto r = converter->convert(m_module.get());
        (void)r;
        assert(succeeded(r) && "second conversion failed");

        auto llvmModule = mlir::translateModuleToLLVMIR(*m_module);
        if (!llvmModule)
        {
            llvm::errs() << "Failed to emit LLVM IR\n";
            return;
        }
        // Initialize LLVM targets.
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());
        auto optPipeline = mlir::makeOptimizingTransformer(/* optLevel=*/3, /* sizeLevel=*/0);
        if (auto err = optPipeline(llvmModule.get()))
        {
            llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
            return;
        }
        if (std::getenv("NGRAPH_MLIR_DUMP_ALL") != nullptr)
        {
            llvmModule->dump();
        }
        return;
    }

// MLIR builders
#define TI(x) std::type_index(typeid(x))

    void MLIRCompiler::build_ng_dialect()
    {
        // TODO: subgraph_topological_sort expects a list of shared_ptr. CPU BE has raw pointers.
        // Fix this.
        //for (auto node : subgraph_topological_sort(m_sub_graph))
        NGRAPH_ASSERT(m_sub_graph.size() == 1) << "Supporting code-gen for a single node for now";
        {
            auto np = m_sub_graph[0];

            auto it = op_dispatcher.find(TI(*np));
            if (it == op_dispatcher.end())
            {
                throw unsupported_op{
                    std::string{"The MLIR backend doesn't currently implement the '"} +
                    np->description() + "' operation"};
            }
            mlir::Value* mlir_value = it->second(*this, np);
            // builders that have multiple result values will update the value map, and set their ret values to null
            if (mlir_value)
            {
                update_tensor_value(np->get_output_tensor_ptr().get(), mlir_value);
            }
        }
        create_return();
    }

    template <>
    mlir::Value* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Add)
    {
        return compiler.create_binary_op<NG_AddOp>(ng_node);
    }

    const MLIRCompiler::MLIRCompOpMap MLIRCompiler::op_dispatcher{
        {TI(ngraph::op::Add), &MLIRCompiler::create_op<ngraph::op::Add>}};

    template <typename BinOp>
    mlir::Value* MLIRCompiler::create_binary_op(const ngraph::Node* ng_node)
    {
        auto lhs = ng_node->get_argument(0)->get_output_tensor_ptr();
        auto rhs = ng_node->get_argument(1)->get_output_tensor_ptr();
        auto lhs_v = get_tensor_value(lhs.get()).m_value;
        auto rhs_v = get_tensor_value(rhs.get()).m_value;
        return m_builder->create<BinOp>(mlir::UnknownLoc::get(&m_context), lhs_v, rhs_v)
            .getResult();
    }

    void MLIRCompiler::create_return()
    {
        std::vector<mlir::Value*> value_list;
        for (auto tensor : m_op_tensors)
        {
            value_list.push_back(get_tensor_value(tensor).m_value);
        }
        m_builder->create<NG_ReturnOp>(mlir::UnknownLoc::get(&m_context), value_list);
    }
}
