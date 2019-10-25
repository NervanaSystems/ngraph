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
/// \file
/// This file contains the pre-generated source code for CPURuntimeContextCG. This class is used
/// to hold runtime information of the execution of kernels in codegen mode.
///

#pragma once

R"(
                enum class OpType
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

    CPURuntimeContextCG() { init_tbb(); init_mkldnn_primitives();}
    ~CPURuntimeContextCG() { cleanup_tbb(); cleanup_mkldnn_primitives();}
#else
    CPURuntimeContextCG() { init_mkldnn_primitives();}
    ~CPURuntimeContextCG() { cleanup_mkldnn_primitives();}
#endif

    std::vector<mkldnn::memory*> mkldnn_memories;
    std::vector<mkldnn::primitive*> mkldnn_primitives;
    std::vector<mkldnn::memory::desc*> mkldnn_scratchpad_mds;
    AlignedBuffer* scratchpad_buffer;
    std::vector<char*> mkldnn_workspaces;
    std::vector<mkldnn::memory::desc*> mkldnn_descriptors;

    mkldnn::engine global_cpu_engine = mkldnn::engine(mkldnn::engine::kind::cpu, 0);

    void set_memory_ptr(size_t index,
                        void* ptr)
	{
		auto memory = mkldnn_memories[index];
		memory->set_data_handle(ptr);
	}

    void mkldnn_invoke_primitive(size_t primitive_index, std::vector<size_t>& deps,
                                        OpType type, size_t scratchpad_size)
	{
        std::unordered_map<int, mkldnn::memory> exec_args;
        size_t nargs;
        switch (type)
        {
        case OpType::ADD:
            exec_args = {{MKLDNN_ARG_MULTIPLE_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_MULTIPLE_SRC + 1, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[2]]}};
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
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[1]]}};
            break;
        case OpType::AVGPOOLBACKPROP:
            exec_args = {{MKLDNN_ARG_DIFF_DST, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_DIFF_SRC, *mkldnn_memories[deps[1]]}};
            break;
        case OpType::BATCHNORM3ARGS:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_WEIGHTS, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[2]]},
                         {MKLDNN_ARG_MEAN, *mkldnn_memories[deps[3]]},
                         {MKLDNN_ARG_VARIANCE, *mkldnn_memories[deps[4]]}};
            break;
        case OpType::BATCHNORM5ARGS:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_MEAN, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_VARIANCE, *mkldnn_memories[deps[2]]},
                         {MKLDNN_ARG_WEIGHTS, *mkldnn_memories[deps[3]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[4]]}};
            break;
        case OpType::BATCHNORMBACKPROP:
            exec_args = {{MKLDNN_ARG_WEIGHTS, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_SRC, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_MEAN, *mkldnn_memories[deps[2]]},
                         {MKLDNN_ARG_VARIANCE, *mkldnn_memories[deps[3]]},
                         {MKLDNN_ARG_DIFF_DST, *mkldnn_memories[deps[4]]},
                         {MKLDNN_ARG_DIFF_SRC, *mkldnn_memories[deps[5]]},
                         {MKLDNN_ARG_DIFF_WEIGHTS, *mkldnn_memories[deps[6]]}};
            break;
        case OpType::CONCAT:
        case OpType::QUANTIZEDCONCAT:
            nargs = deps.size() - 1;
            for (size_t i = 0; i < nargs; i++)
            {
                exec_args.insert({MKLDNN_ARG_MULTIPLE_SRC + i, *mkldnn_memories[deps[i]]});
            }
            exec_args.insert({MKLDNN_ARG_DST, *mkldnn_memories[deps[nargs]]});
            break;
        case OpType::CONVOLUTION:
        case OpType::CONVOLUTIONRELU:
        case OpType::CONVOLUTIONADD:
        case OpType::GROUPCONVOLUTION:
        case OpType::QUANTIZEDMATMUL:
        case OpType::QUANTIZEDCONVOLUTION:
        case OpType::QUANTIZEDCONVOLUTIONRELU:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_WEIGHTS, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[2]]}};
            break;
        case OpType::CONVOLUTIONBIAS:
        case OpType::CONVOLUTIONBIASADD:
        case OpType::GROUPCONVOLUTIONBIAS:
        case OpType::QUANTIZEDCONVOLUTIONBIAS:
        case OpType::QUANTIZEDCONVOLUTIONBIASADD:
        case OpType::QUANTIZEDCONVOLUTIONBIASSIGNEDADD:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_WEIGHTS, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_BIAS, *mkldnn_memories[deps[2]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[3]]}};
            break;
        case OpType::QUANTIZEDDOTBIAS:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_WEIGHTS, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_BIAS, *mkldnn_memories[deps[3]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[2]]}};
            break;
        case OpType::CONVOLUTIONBACKPROPDATA:
            exec_args = {{MKLDNN_ARG_DIFF_DST, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_WEIGHTS, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_DIFF_SRC, *mkldnn_memories[deps[2]]}};
            break;
        case OpType::CONVOLUTIONBACKPROPWEIGHTS:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_DIFF_DST, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_DIFF_WEIGHTS, *mkldnn_memories[deps[2]]}};
            break;
        case OpType::CONVOLUTIONBIASBACKPROPWEIGHTSBIAS:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_DIFF_DST, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_DIFF_WEIGHTS, *mkldnn_memories[deps[2]]},
                         {MKLDNN_ARG_DIFF_BIAS, *mkldnn_memories[deps[3]]}};
            break;
        case OpType::DECONVOLUTIONBIAS:
            exec_args = {{MKLDNN_ARG_WEIGHTS, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_SRC, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_BIAS, *mkldnn_memories[deps[2]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[3]]}};
            break;
        case OpType::LSTM:
        case OpType::RNN:
            exec_args = {{MKLDNN_ARG_SRC_LAYER, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_SRC_ITER, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_SRC_ITER_C, *mkldnn_memories[deps[2]]},
                         {MKLDNN_ARG_WEIGHTS_LAYER, *mkldnn_memories[deps[3]]},
                         {MKLDNN_ARG_WEIGHTS_ITER, *mkldnn_memories[deps[4]]},
                         {MKLDNN_ARG_BIAS, *mkldnn_memories[deps[5]]},
                         {MKLDNN_ARG_DST_LAYER, *mkldnn_memories[deps[6]]},
                         {MKLDNN_ARG_DST_ITER, *mkldnn_memories[deps[7]]},
                         {MKLDNN_ARG_DST_ITER_C, *mkldnn_memories[deps[8]]},
                         {MKLDNN_ARG_WORKSPACE, *mkldnn_memories[deps[9]]}};
            break;
        case OpType::MAXPOOLBACKPROPFORWARD:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_WORKSPACE, *mkldnn_memories[deps[3]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[2]]}};
            break;
        case OpType::MAXPOOLWITHINDICES:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_WORKSPACE, *mkldnn_memories[deps[2]]},
                         {MKLDNN_ARG_DST, *mkldnn_memories[deps[1]]}};
            break;
        case OpType::MAXPOOLBACKPROPBACKWARD:
            exec_args = {{MKLDNN_ARG_DIFF_DST, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_WORKSPACE, *mkldnn_memories[deps[3]]},
                         {MKLDNN_ARG_DIFF_SRC, *mkldnn_memories[deps[2]]}};
		    break;
        case OpType::MAXPOOLWITHINDICESBACKPROP:
            exec_args = {{MKLDNN_ARG_DIFF_DST, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_WORKSPACE, *mkldnn_memories[deps[2]]},
                         {MKLDNN_ARG_DIFF_SRC, *mkldnn_memories[deps[1]]}};
            break;
        case OpType::RELUBACKPROP:
        case OpType::SIGMOIDBACKPROP:
            exec_args = {{MKLDNN_ARG_SRC, *mkldnn_memories[deps[0]]},
                         {MKLDNN_ARG_DIFF_DST, *mkldnn_memories[deps[1]]},
                         {MKLDNN_ARG_DIFF_SRC, *mkldnn_memories[deps[2]]}};
            break;
        }

        if (scratchpad_size)
        {
            mkldnn::memory scratchpad(*mkldnn_scratchpad_mds[primitive_index],
                                      global_cpu_engine,
                                      scratchpad_buffer->get_ptr());
            exec_args.insert({MKLDNN_ARG_SCRATCHPAD, scratchpad});
        }

        mkldnn::stream s(global_cpu_engine);
        try
        {
            (*mkldnn_primitives[primitive_index]).execute(s, exec_args);
            s.wait();
        }
        catch (const mkldnn::error& e)
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

    void init_mkldnn_primitives();

	inline void cleanup_mkldnn_primitives()
	{
		for (auto p : mkldnn_primitives)
		{
			delete p;
		}
	    for (auto m : mkldnn_memories)
		{
			delete m;
		}
        for (auto s : mkldnn_scratchpad_mds)
        {
            delete s;
        }
        delete scratchpad_buffer;
		
#ifndef _WIN32
        //To avoid memory leak in mkldnn, release any buffers that are not free'd yet.
        //https://software.intel.com/en-us/mkl-linux-developer-guide-avoiding-memory-leaks-in-intel-mkl
        //mkl_free_buffers() is not exposed at this point, hence using mkl_serv_free_buffers()
        ngraph::runtime::cpu::mkldnn_utils::mkl_serv_free_buffers();
#endif

        for (auto w : mkldnn_workspaces)
        {
            free(w);
        }
    }

    inline void cleanup_mkldnn_descriptors()
    {
        for (auto d : mkldnn_descriptors)
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
    cg_ctx->mkldnn_descriptors = std::vector<mkldnn::memory::desc*>(descs_count);
    for (auto i = 0; i < descs_count; i++)
    {
    		size_t index;
		    desc_file >> index;
        auto desc = (mkldnn::memory::desc*)malloc(sizeof(mkldnn::memory::desc));
        if (!desc)
        {
            throw std::bad_alloc();
        }
        desc_file.read(reinterpret_cast<char*>(desc), sizeof(mkldnn::memory::desc));

		    cg_ctx->mkldnn_descriptors[i] = desc;
		    cg_ctx->mkldnn_memories[index] = new mkldnn::memory(*cg_ctx->mkldnn_descriptors[i], cg_ctx->global_cpu_engine, nullptr);
	}
};
)"
