// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <exception>
#include <sstream>

#include "log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/assign_tensors.hpp"
#include "ngraph/pass/liveness.hpp"

using namespace std;
using namespace ngraph;

bool pass::Liveness::run_on_call_list(list<Node*>& ops)
{
    // list<Node*> live_list;
    // list<Node*> free_list;
    // list<Node*> new_list;
    // currently_live = list();

    // size_t i = 0;
    // for (i, exop in enumerate(reversed(ops)
    // for(auto it=ops.rbegin(); it!=ops.rend(); it++)
    // {
    //     Node& exop = **it;
    //     input_tensor_decls = list()
    //     for (auto input_decl : exop.get_inputs())
    //     {
    //         if (is_interesting(input_decl.tensor_decl))
    //         {
    //             input_tensor_decls.append(input_decl.tensor_decl);
    //         }
    //     }

    //     output_tensor_decls = list()
    //     for (output_decl : exop.output_decls)
    //     {
    //         if (is_interesting(output_decl.tensor_decl))
    //         {
    //             output_tensor_decls.append(output_decl.tensor_decl);
    //         }
    //     }

    //     free_tensor_decls = list();
    //     new_tensor_decls = list();
    //     for tensor_decl in input_tensor_decls + output_tensor_decls
    //     {
    //         if tensor_decl not in currently_live
    //         {
    //             // this is the last node that value is seen in
    //             // delete it at the end of the op
    //             currently_live.append(tensor_decl);
    //             free_tensor_decls.append(tensor_decl);
    //         }
    //     }
    //     live_list.insert(0, list(currently_live))
    //     for output_decl in output_tensor_decls
    //     {
    //         if output_decl in currently_live
    //         {
    //             new_tensor_decls.append(output_decl);
    //             currently_live.remove(output_decl);
    //         }
    //     }
    //     free_list.insert(0, free_tensor_decls);
    //     new_list.insert(0, new_tensor_decls);
    // }

    // // Anything marked as output must remain live for the remainder of the graph
    // // Add outputs to live_list and remove from free_list
    // outputs = list();
    // seen = list();
    // for i, exop in enumerate(ops)
    // {
    //     for tensor in live_list[i]
    //     {
    //         if tensor.is_output and tensor not in outputs
    //         {
    //             outputs.append(tensor);
    //         }
    //     }
    //     for tensor in outputs
    //     {
    //         if tensor not in live_list[i]
    //         {
    //             live_list[i].append(tensor);
    //         }
    //         if tensor in free_list[i]
    //         {
    //             free_list[i].remove(tensor);
    //         }
    //         if tensor in new_list[i]
    //         {
    //             if tensor in seen
    //             {
    //                 new_list[i].remove(tensor);
    //             }
    //             else
    //             {
    //                 seen.append(tensor);
    //             }
    //         }
    //     }
    //     exop.liveness_live_list = live_list[i];
    //     exop.liveness_new_list = new_list[i];
    //     exop.liveness_free_list = free_list[i];
    // }

    // self.validate_liveness(ops)
    return false;
}

void pass::Liveness::check_dependencies(
    const std::vector<std::shared_ptr<CallBase>>& registered_passes) const
{
    bool found_propagate_types = false;
    for (auto pass : registered_passes)
    {
        if (dynamic_pointer_cast<AssignTensors>(pass))
        {
            found_propagate_types = true;
        }
    }

    if (!found_propagate_types)
    {
        throw runtime_error("Depencency 'PropagateTypes' not found for pass 'AssignTensors'");
    }
}

// bool pass::Liveness::is_interesting(tensor_decl)
// {
//     return
//         tensor_decl.is_persistent == false &&
//         tensor_decl.is_constant == false &&
//         tensor_decl.is_compile_only == false;
// }

// void pass::Liveness::validate_liveness(ops)
// {
//     dead_tensors = set();
//     for i, exop in enumerate(ops)
//     {
//         active = set(exop.liveness_live_list);
//         active |= set(exop.liveness_new_list);
//         active |= set(exop.liveness_free_list);
//         if bool(dead_tensors.intersection(active)) is True
//         {
//             raise RuntimeError("Liveness: Dead tensors intersect active tensors");
//         }
//         for tensor in exop.liveness_free_list
//         {
//             dead_tensors.add(tensor);
//         }
//     }
// }

