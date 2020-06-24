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
/// \file
/// This file contains the pre-generated source code for CPURuntimeContextCG. This class is used
/// to hold runtime information of the execution of kernels in codegen mode.
///

#pragma once

R"(enum class OpType
{
    ADD,
    AVGPOOL,
    AVGPOOLBACKPROP,
    BATCHNORM3ARGS,
    BATCHNORM5ARGS,
    BATCHNORMBACKPROP,
    BOUNDEDRELU,
    CONCAT,
    CONVERTLAYOUT,
    CONVOLUTION,
    CONVOLUTIONRELU,
    CONVOLUTIONADD,
    CONVOLUTIONBIAS,
    CONVOLUTIONBIASADD,
    CONVOLUTIONBACKPROPDATA,
    CONVOLUTIONBACKPROPWEIGHTS,
    CONVOLUTIONBIASBACKPROPWEIGHTSBIAS,
    GROUPCONVOLUTION,
    GROUPCONVOLUTIONBIAS,
    DECONVOLUTIONBIAS,
    LEAKYRELU,
    LRN,
    LSTM,
    MAXPOOL,
    MAXPOOLBACKPROPFORWARD,
    MAXPOOLBACKPROPBACKWARD,
    MAXPOOLWITHINDICES,
    MAXPOOLWITHINDICESBACKPROP,
    QUANTIZE,
    DEQUANTIZE,
    QUANTIZEDAVGPOOL,
    QUANTIZEDMAXPOOL,
    QUANTIZEDCONCAT,
    QUANTIZEDDOTBIAS,
    QUANTIZEDMATMUL,
    QUANTIZEDCONVOLUTION,
    QUANTIZEDCONVOLUTIONBIAS,
    QUANTIZEDCONVOLUTIONBIASADD,
    QUANTIZEDCONVOLUTIONBIASSIGNEDADD,
    QUANTIZEDCONVOLUTIONRELU,
    RELU,
    RELUBACKPROP,
    RNN,
    SIGMOID,
    SIGMOIDBACKPROP,
    SLICE,
    SOFTMAX
};

struct CPURuntimeContextCG
{
#if defined(NGRAPH_TBB_ENABLE)
    std::unique_ptr<tbb::flow::graph> tbb_graph;
    std::unique_ptr<tbb::global_control> tbb_gcontrol;

    CPURuntimeContextCG() { init_tbb(); init_dnnl_primitives();}
    ~CPURuntimeContextCG() { cleanup_tbb(); cleanup_dnnl_primitives();}
#else
    CPURuntimeContextCG() { init_dnnl_primitives();}
    ~CPURuntimeContextCG() { cleanup_dnnl_primitives();}
#endif

    std::vector<dnnl::memory*> dnnl_memories;
    std::vector<dnnl::primitive*> dnnl_primitives;
    std::vector<dnnl::memory::desc*> dnnl_scratchpad_mds;
    AlignedBuffer* scratchpad_buffer;
    std::vector<char*> dnnl_workspaces;
    std::vector<dnnl::memory::desc*> dnnl_descriptors;

    dnnl::engine global_cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);

    void set_memory_ptr(size_t index,
                        void* ptr)
    {
        auto memory = dnnl_memories[index];
        memory->set_data_handle(ptr);
    }

    void dnnl_invoke_primitive(size_t primitive_index, std::vector<size_t>& deps,
                                        OpType type, size_t scratchpad_size)
    {
        std::unordered_map<int, dnnl::memory> exec_args;
        size_t nargs;
        switch (type)
        {
        case OpType::ADD:
            exec_args = {{DNNL_ARG_MULTIPLE_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_MULTIPLE_SRC + 1, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[2]]}};
            break;
        case OpType::AVGPOOL:
        case OpType::BOUNDEDRELU:
        case OpType::CONVERTLAYOUT:
        case OpType::LEAKYRELU:
        case OpType::LRN:
        case OpType::MAXPOOL:
        case OpType::QUANTIZE:
        case OpType::DEQUANTIZE:
        case OpType::QUANTIZEDAVGPOOL:
        case OpType::QUANTIZEDMAXPOOL:
        case OpType::RELU:
        case OpType::SIGMOID:
        case OpType::SLICE:
        case OpType::SOFTMAX:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[1]]}};
            break;
        case OpType::AVGPOOLBACKPROP:
            exec_args = {{DNNL_ARG_DIFF_DST, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_DIFF_SRC, *dnnl_memories[deps[1]]}};
            break;
        case OpType::BATCHNORM3ARGS:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_WEIGHTS, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[2]]},
                         {DNNL_ARG_MEAN, *dnnl_memories[deps[3]]},
                         {DNNL_ARG_VARIANCE, *dnnl_memories[deps[4]]}};
            break;
        case OpType::BATCHNORM5ARGS:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_MEAN, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_VARIANCE, *dnnl_memories[deps[2]]},
                         {DNNL_ARG_WEIGHTS, *dnnl_memories[deps[3]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[4]]}};
            break;
        case OpType::BATCHNORMBACKPROP:
            exec_args = {{DNNL_ARG_WEIGHTS, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_SRC, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_MEAN, *dnnl_memories[deps[2]]},
                         {DNNL_ARG_VARIANCE, *dnnl_memories[deps[3]]},
                         {DNNL_ARG_DIFF_DST, *dnnl_memories[deps[4]]},
                         {DNNL_ARG_DIFF_SRC, *dnnl_memories[deps[5]]},
                         {DNNL_ARG_DIFF_WEIGHTS, *dnnl_memories[deps[6]]}};
            break;
        case OpType::CONCAT:
        case OpType::QUANTIZEDCONCAT:
            nargs = deps.size() - 1;
            for (size_t i = 0; i < nargs; i++)
            {
                exec_args.insert({DNNL_ARG_MULTIPLE_SRC + i, *dnnl_memories[deps[i]]});
            }
            exec_args.insert({DNNL_ARG_DST, *dnnl_memories[deps[nargs]]});
            break;
        case OpType::CONVOLUTION:
        case OpType::CONVOLUTIONRELU:
        case OpType::CONVOLUTIONADD:
        case OpType::GROUPCONVOLUTION:
        case OpType::QUANTIZEDMATMUL:
        case OpType::QUANTIZEDCONVOLUTION:
        case OpType::QUANTIZEDCONVOLUTIONRELU:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_WEIGHTS, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[2]]}};
            break;
        case OpType::CONVOLUTIONBIAS:
        case OpType::CONVOLUTIONBIASADD:
        case OpType::GROUPCONVOLUTIONBIAS:
        case OpType::QUANTIZEDCONVOLUTIONBIAS:
        case OpType::QUANTIZEDCONVOLUTIONBIASADD:
        case OpType::QUANTIZEDCONVOLUTIONBIASSIGNEDADD:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_WEIGHTS, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_BIAS, *dnnl_memories[deps[2]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[3]]}};
            break;
        case OpType::QUANTIZEDDOTBIAS:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_WEIGHTS, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_BIAS, *dnnl_memories[deps[3]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[2]]}};
            break;
        case OpType::CONVOLUTIONBACKPROPDATA:
            exec_args = {{DNNL_ARG_DIFF_DST, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_WEIGHTS, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_DIFF_SRC, *dnnl_memories[deps[2]]}};
            break;
        case OpType::CONVOLUTIONBACKPROPWEIGHTS:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_DIFF_DST, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_DIFF_WEIGHTS, *dnnl_memories[deps[2]]}};
            break;
        case OpType::CONVOLUTIONBIASBACKPROPWEIGHTSBIAS:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_DIFF_DST, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_DIFF_WEIGHTS, *dnnl_memories[deps[2]]},
                         {DNNL_ARG_DIFF_BIAS, *dnnl_memories[deps[3]]}};
            break;
        case OpType::DECONVOLUTIONBIAS:
            exec_args = {{DNNL_ARG_WEIGHTS, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_SRC, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_BIAS, *dnnl_memories[deps[2]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[3]]}};
            break;
        case OpType::LSTM:
        case OpType::RNN:
            exec_args = {{DNNL_ARG_SRC_LAYER, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_SRC_ITER, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_SRC_ITER_C, *dnnl_memories[deps[2]]},
                         {DNNL_ARG_WEIGHTS_LAYER, *dnnl_memories[deps[3]]},
                         {DNNL_ARG_WEIGHTS_ITER, *dnnl_memories[deps[4]]},
                         {DNNL_ARG_BIAS, *dnnl_memories[deps[5]]},
                         {DNNL_ARG_DST_LAYER, *dnnl_memories[deps[6]]},
                         {DNNL_ARG_DST_ITER, *dnnl_memories[deps[7]]},
                         {DNNL_ARG_DST_ITER_C, *dnnl_memories[deps[8]]},
                         {DNNL_ARG_WORKSPACE, *dnnl_memories[deps[9]]}};
            break;
        case OpType::MAXPOOLBACKPROPFORWARD:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_WORKSPACE, *dnnl_memories[deps[3]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[2]]}};
            break;
        case OpType::MAXPOOLWITHINDICES:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_WORKSPACE, *dnnl_memories[deps[2]]},
                         {DNNL_ARG_DST, *dnnl_memories[deps[1]]}};
            break;
        case OpType::MAXPOOLBACKPROPBACKWARD:
            exec_args = {{DNNL_ARG_DIFF_DST, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_WORKSPACE, *dnnl_memories[deps[3]]},
                         {DNNL_ARG_DIFF_SRC, *dnnl_memories[deps[2]]}};
            break;
        case OpType::MAXPOOLWITHINDICESBACKPROP:
            exec_args = {{DNNL_ARG_DIFF_DST, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_WORKSPACE, *dnnl_memories[deps[2]]},
                         {DNNL_ARG_DIFF_SRC, *dnnl_memories[deps[1]]}};
            break;
        case OpType::RELUBACKPROP:
        case OpType::SIGMOIDBACKPROP:
            exec_args = {{DNNL_ARG_SRC, *dnnl_memories[deps[0]]},
                         {DNNL_ARG_DIFF_DST, *dnnl_memories[deps[1]]},
                         {DNNL_ARG_DIFF_SRC, *dnnl_memories[deps[2]]}};
            break;
        }

        if (scratchpad_size)
        {
            dnnl::memory scratchpad(*dnnl_scratchpad_mds[primitive_index],
                                      global_cpu_engine,
                                      scratchpad_buffer->get_ptr());
            exec_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
        }

        dnnl::stream s(global_cpu_engine);
        try
        {
            (*dnnl_primitives[primitive_index]).execute(s, exec_args);
            s.wait();
        }
        catch (const dnnl::error& e)
        {
            throw std::runtime_error("Could not run mkdnn primitive " + std::string(e.message));
        }
    }

private:
#if defined(NGRAPH_TBB_ENABLE)
    inline void init_tbb()
    {
        if (std::getenv("NGRAPH_CPU_USE_TBB"))
        {
            tbb_graph.reset(new tbb::flow::graph);
            const char* env_parallelism = std::getenv("NGRAPH_INTER_OP_PARALLELISM");
            const int parallelism = env_parallelism == nullptr ? 1 : std::atoi(env_parallelism);
            tbb_gcontrol.reset(
                new tbb::global_control(tbb::global_control::max_allowed_parallelism, parallelism));
        }
    }

    inline void cleanup_tbb()
    {
        if (std::getenv("NGRAPH_CPU_USE_TBB"))
        {
            // Delete nodes in tbb_graph.
            tbb_graph->wait_for_all();
            std::vector<tbb::flow::graph_node*> to_be_deleted;
            for (auto it = tbb_graph->begin(); it != tbb_graph->end(); it++)
            {
                to_be_deleted.push_back(&*it);
            }
            for (auto* node : to_be_deleted)
            {
                delete node;
            }
        }
    }
#endif

    void init_dnnl_primitives();

    inline void cleanup_dnnl_primitives()
    {
        for (auto p : dnnl_primitives)
        {
            delete p;
        }
        for (auto m : dnnl_memories)
        {
            delete m;
        }
        for (auto s : dnnl_scratchpad_mds)
        {
            delete s;
        }
        delete scratchpad_buffer;
        
#ifndef _WIN32
        //To avoid memory leak in dnnl, release any buffers that are not free'd yet.
        //https://software.intel.com/en-us/mkl-linux-developer-guide-avoiding-memory-leaks-in-intel-mkl
        //mkl_free_buffers() is not exposed at this point, hence using mkl_serv_free_buffers()
        ngraph::runtime::cpu::dnnl_utils::mkl_serv_free_buffers();
#endif

        for (auto w : dnnl_workspaces)
        {
            free(w);
        }

        for (auto d : dnnl_descriptors)
        {
            free(d);
        }
    }
};

extern "C" CPURuntimeContextCG* init_cg_ctx()
{
    return new CPURuntimeContextCG;
}

extern "C" void destroy_cg_ctx(CPURuntimeContextCG* cg_ctx)
{
    delete cg_ctx;
}

static void
    deserialize_memory_descs_and_build_memory(std::ifstream& desc_file,
                                              CPURuntimeContextCG* cg_ctx,
                                              size_t descs_count)
{
    cg_ctx->dnnl_descriptors = std::vector<dnnl::memory::desc*>(descs_count);
    for (auto i = 0; i < descs_count; i++)
    {
        size_t index;
        desc_file >> index;
        auto desc = (dnnl::memory::desc*)malloc(sizeof(dnnl::memory::desc));
        if (!desc)
        {
            throw std::bad_alloc();
        }
        desc_file.read(reinterpret_cast<char*>(desc), sizeof(dnnl::memory::desc));

        cg_ctx->dnnl_descriptors[i] = desc;
        cg_ctx->dnnl_memories[index] = new dnnl::memory(*cg_ctx->dnnl_descriptors[i], cg_ctx->global_cpu_engine, nullptr);
    }
};
)"
