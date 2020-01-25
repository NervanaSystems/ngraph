//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License")
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

// This collection contains one entry for each environment variable.If an environment variable is
// added it must be added to this list.
//
// In order to use this list you want to define a macro named exactly NGRAPH_DEFINE_ENVVAR
// When you are done you should undef the macro

#ifndef NGRAPH_DEFINE_ENVVAR
#warning "NGRAPH_DEFINE_ENVVAR not defined"
#define NGRAPH_DEFINE_ENVVAR(ENUMID, NAME, DEFAULT, DESCRIPTION)
#endif

NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CODEGEN,
                     "NGRAPH_CODEGEN",
                     "FALSE",
                     "Enable ngraph codegen")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_COMPILER_DEBUGINFO_ENABLE,
                     "NGRAPH_COMPILER_DEBUGINFO_ENABLE",
                     "FALSE",
                     "Enable compiler debug info when codegen is enabled")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_COMPILER_DIAG_ENABLE,
                     "NGRAPH_COMPILER_DIAG_ENABLE",
                     "FALSE",
                     "Enable compiler diagnostics when codegen is enabled")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_COMPILER_REPORT_ENABLE,
                     "NGRAPH_COMPILER_REPORT_ENABLE",
                     "FALSE",
                     "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_BIN_TRACER_LOG,
                     "NGRAPH_CPU_BIN_TRACER_LOG",
                     "",
                     "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_CHECK_PARMS_AND_CONSTS,
                     "NGRAPH_CPU_CHECK_PARMS_AND_CONSTS",
                     "FALSE",
                     "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_CONCURRENCY, "NGRAPH_CPU_CONCURRENCY", "1", "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_DEBUG_TRACER,
                     "NGRAPH_CPU_DEBUG_TRACER",
                     "FALSE",
                     "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_EIGEN_THREAD_COUNT,
                     "NGRAPH_CPU_EIGEN_THREAD_COUNT",
                     "0",
                     "Calculated in code")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_INF_CHECK, "NGRAPH_CPU_INF_CHECK", "FALSE", "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_NAN_CHECK, "NGRAPH_CPU_NAN_CHECK", "FALSE", "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_TRACER_LOG, "NGRAPH_CPU_TRACER_LOG", "", "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_TRACING,
                     "NGRAPH_CPU_TRACING",
                     "FALSE",
                     "Generates timelines to view in chrome://tracing when enabled")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_USE_REF_KERNELS,
                     "NGRAPH_CPU_USE_REF_KERNELS",
                     "FALSE",
                     "Use reference kernels instead of specialized kernels")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_CPU_USE_TBB,
                     "NGRAPH_CPU_USE_TBB",
                     "FALSE",
                     "Enable use of TBB. Experimental")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_DECONV_FUSE, "NGRAPH_DECONV_FUSE", "FALSE", "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_DEX_DEBUG,
                     "NGRAPH_DEX_DEBUG",
                     "FALSE",
                     "Generates debug info for direct execution(DEX) mode of ngraph")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_DISABLE_LOGGING,
                     "NGRAPH_DISABLE_LOGGING",
                     "FALSE",
                     "Disable printing all logs irrespective of build type")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_DISABLED_FUSIONS,
                     "NGRAPH_DISABLED_FUSIONS",
                     "",
                     "Disable specified fusions. Specified as  separated list and supports regex")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_ENABLE_REPLACE_CHECK,
                     "NGRAPH_ENABLE_REPLACE_CHECK",
                     "FALSE",
                     "Enables strict type checking in copy constructor copy_with_new_args")
NGRAPH_DEFINE_ENVVAR(
    ngraph::EnvVarEnum::NGRAPH_ENABLE_SERIALIZE_TRACING,
    "NGRAPH_ENABLE_SERIALIZE_TRACING",
    "FALSE",
    "Enables creating serialized json files to be run with nbench. Generates 1 file per pass")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_ENABLE_TRACING,
                     "NGRAPH_ENABLE_TRACING",
                     "FALSE",
                     "Enables creating graph execution timelines to be viewed in chrome://tracing")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_ENABLE_VISUALIZE_TRACING,
                     "NGRAPH_ENABLE_VISUALIZE_TRACING",
                     "FALSE",
                     "Enables creating visual graph for each pass. By default generates .svg files")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_FAIL_MATCH_AT,
                     "NGRAPH_FAIL_MATCH_AT",
                     "",
                     "Allows one to specify node name patterns to abort pattern matching at "
                     "particular nodes. Helps debug an offending fusion")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_GRAPH_REWRITE_RERUN_DYNAMIC_CHECK,
                     "NGRAPH_GRAPH_REWRITE_RERUN_DYNAMIC_CHECK",
                     "FALSE",
                     "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_GTEST_INFO,
                     "NGRAPH_GTEST_INFO",
                     "FALSE",
                     "Enables printing info about a specific test")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_INTER_OP_PARALLELISM,
                     "NGRAPH_INTER_OP_PARALLELISM",
                     "1",
                     "Flag to control performance on Xeon/CPU backend")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_INTRA_OP_PARALLELISM,
                     "NGRAPH_INTRA_OP_PARALLELISM",
                     "1",
                     "Flag to control performance on Xeon/CPU backend")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_MLIR,
                     "NGRAPH_MLIR",
                     "FALSE",
                     "Flag to enable MLIR in ngraph")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_MLIR_MAX_CYCLE_DEPTH,
                     "NGRAPH_MLIR_MAX_CYCLE_DEPTH",
                     "20",
                     "Max cycle depth in MLIR")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_MLIR_OPT_LEVEL,
                     "NGRAPH_MLIR_OPT_LEVEL",
                     "",
                     "Define optimization level in MLIR")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_MLIR_OPTIONS,
                     "NGRAPH_MLIR_OPTIONS",
                     "",
                     "Define MLIR options")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_PASS_ATTRIBUTES,
                     "NGRAPH_PASS_ATTRIBUTES",
                     "",
                     "Specify pass specific attributes to be enabled or disabled. specify a "
                     "semi-colon separated list. Naming of pass attributes is up to the backends.")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_PASS_CPU_LAYOUT_ELTWISE,
                     "NGRAPH_PASS_CPU_LAYOUT_ELTWISE",
                     "0",
                     "")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_PASS_ENABLES,
                     "NGRAPH_PASS_ENABLES",
                     "",
                     "Specify a semi-colon separated list to enable or disable a pass. This will "
                     "override the default enable/disable values")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_PROFILE_PASS_ENABLE,
                     "NGRAPH_PROFILE_PASS_ENABLE",
                     "FALSE",
                     "Dump the name and execution time of each pass")
NGRAPH_DEFINE_ENVVAR(
    ngraph::EnvVarEnum::NGRAPH_PROVENANCE_ENABLE,
    "NGRAPH_PROVENANCE_ENABLE",
    "FALSE",
    "Enable adding provenance info to nodes. This will also be added to serialized files.")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_SERIALIZER_OUTPUT_SHAPES,
                     "NGRAPH_SERIALIZER_OUTPUT_SHAPES",
                     "FALSE",
                     "Enable adding output shapes in the serialized graph")
NGRAPH_DEFINE_ENVVAR(
    ngraph::EnvVarEnum::NGRAPH_VISUALIZE_EDGE_JUMP_DISTANCE,
    "NGRAPH_VISUALIZE_EDGE_JUMP_DISTANCE",
    "FALSE",
    "Calculated in code helps prevent *long* edges between two nodes very far apart")
NGRAPH_DEFINE_ENVVAR(
    ngraph::EnvVarEnum::NGRAPH_VISUALIZE_EDGE_LABELS,
    "NGRAPH_VISUALIZE_EDGE_LABELS",
    "FALSE",
    "When enabled, adds label to a graph edge when NGRAPH_ENABLE_VISUALIZE_TRACING=1")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::NGRAPH_VISUALIZE_TRACING_FORMAT,
                     "NGRAPH_VISUALIZE_TRACING_FORMAT",
                     "",
                     "Default file format is .svg, one can change it to pdf or png use this ")
NGRAPH_DEFINE_ENVVAR(
    ngraph::EnvVarEnum::NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES,
    "NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES",
    "FALSE",
    "When enabled, add output shape of a node when NGRAPH_ENABLE_VISUALIZE_TRACING=1")
NGRAPH_DEFINE_ENVVAR(
    ngraph::EnvVarEnum::NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES,
    "NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES",
    "FALSE",
    "When enabled, add output type of a node when NGRAPH_ENABLE_VISUALIZE_TRACING=1")
NGRAPH_DEFINE_ENVVAR(ngraph::EnvVarEnum::OMP_NUM_THREADS,
                     "OMP_NUM_THREADS",
                     "",
                     "By default it uses the max number of cores available")
