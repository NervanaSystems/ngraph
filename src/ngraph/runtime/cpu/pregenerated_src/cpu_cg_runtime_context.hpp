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
struct CPURuntimeContextCG
{
    std::unique_ptr<tbb::flow::graph> tbb_graph;
    std::unique_ptr<tbb::global_control> tbb_gcontrol;

    CPURuntimeContextCG() { init_tbb(); init_mkldnn_primitives();}
    ~CPURuntimeContextCG() { cleanup_tbb(); cleanup_mkldnn_primitives();}

    std::vector<mkldnn::primitive*> mkldnn_primitives;
    std::vector<char*> mkldnn_workspaces;
	std::vector<mkldnn::memory::desc*> mkldnn_descriptors;

    mkldnn::engine global_cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);

	void set_memory_ptr(size_t primitive_index,
                        void* ptr)
	{
		auto primitive = static_cast<mkldnn::memory*>(mkldnn_primitives[primitive_index]);
		primitive->set_data_handle(ptr);
	}

	void mkldnn_invoke_primitive(size_t primitive_index)
	{
		mkldnn::stream s(mkldnn::stream::kind::eager);
		try
		{
			s.submit({*mkldnn_primitives[primitive_index]}).wait();
		}
		catch (const mkldnn::error& e)
		{
			throw std::runtime_error("Could not run mkldnn primitive " + e.message);
		}
	}


private:
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

    void init_mkldnn_primitives();

	inline void cleanup_mkldnn_primitives()
	{
		for (auto p : mkldnn_primitives)
		{
			delete p;
		}
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
	deserialize_memory_descs_and_build_memory_primitives(std::ifstream& desc_file,
														 CPURuntimeContextCG* cg_ctx,
														 size_t descs_count)
{
	cg_ctx->mkldnn_descriptors = std::vector<mkldnn::memory::desc*>(descs_count);
	for (auto i = 0; i < descs_count; i++)
    {
		size_t primitive_index;
		desc_file >> primitive_index;
        auto desc = (mkldnn::memory::desc*)malloc(sizeof(mkldnn::memory::desc));
		if (!desc)
		{
			throw std::bad_alloc();
		}
        desc_file.read(reinterpret_cast<char*>(desc), sizeof(mkldnn::memory::desc));
		cg_ctx->mkldnn_descriptors[i] = desc;
		cg_ctx->mkldnn_primitives[primitive_index] = new mkldnn::memory({*cg_ctx->mkldnn_descriptors[i], cg_ctx->global_cpu_engine}, nullptr);
	}
};
)"
