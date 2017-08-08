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

#include <memory>
#include <sstream>
#include <exception>
#include <cmath>

#include "exop.hpp"
#include "op_graph.hpp"
#include "util.hpp"

using namespace ngraph;

//================================================================================================
// InputDecl
//================================================================================================

InputDecl::InputDecl(const ExOp&            _exop,
                     size_t                 _pos,
                     tensor_description_ptr _tensor_description,
                     OutputDecl*            _value)
    : exop{_exop}
    , pos{_pos}
    , tensor_description{_tensor_description}
    , read_view{nullptr}
    , m_value{_value}
{
}

TensorDecl& InputDecl::tensor_decl()
{
    return read_view->tensor_decl;
}

OutputDecl* InputDecl::value()
{
    // Returns: The argument supplying this value.
    return m_value;
}

const OutputDecl* InputDecl::value() const
{
    // Returns: The argument supplying this value.
    return m_value;
}

void InputDecl::value(OutputDecl* value)
{
    // Changes the value assigned to this argument, updating value users.

    // Args:
    //     value: The new value for this argument.
    if (m_value != nullptr)
    {
        remove_from(m_value->value_users, this);
        remove_from(read_view->readers, this);
    }
    if (m_value != nullptr)
    {
        tensor_description = value->tensor_description;
    }
    m_value = value;
    if (value != nullptr)
    {
        value->value_users.insert(this);
        read_view = value->write_view()->get_tensor_view(tensor_description, this);
    }
}

std::ostream& ngraph::operator<<(std::ostream& out, const InputDecl& obj)
{
    out << "Arg(" << obj.exop.name() << obj.pos << ")";
    return out;
}

//================================================================================================
// OutputDecl
//================================================================================================

OutputDecl::OutputDecl(const ExOp&            _exop,
                       size_t                 _pos,
                       tensor_decl_ptr        _tensor_decl,
                       tensor_description_ptr _tensor_description)
    : exop{_exop}
    , pos{_pos}
    , tensor_description{_tensor_description}
    , __tensor{_tensor_decl}
    , __write_view{nullptr}
    , value_users{}
{
}

tensor_decl_ptr OutputDecl::tensor_decl()
{
    return __tensor;
}

void OutputDecl::tensor_decl(tensor_decl_ptr tensor_decl)
{
    if (__tensor == tensor_decl)
    {
        return;
    }
    if (__tensor != nullptr)
    {
        tensor_decl->merge_flags(*__tensor);
    }
    __tensor = tensor_decl;
    write_view(tensor_decl->get_tensor_view(tensor_description, this));
}

tensor_view_decl_ptr OutputDecl::write_view()
{
    return __write_view;
}

void OutputDecl::write_view(tensor_view_decl_ptr view)
{
    if (view == nullptr && value_users.size() > 0)
    {
        throw std::runtime_error("Cannot deallocate a view that is in use");
    }
    __write_view = view;
    view->value  = this;
    if (view != nullptr)
    {
        for (InputDecl* arg : value_users)
        {
            arg->tensor_description = tensor_description;
            arg->read_view          = view->get_tensor_view(arg->tensor_description, arg);
        }
    }
}

std::ostream& ngraph::operator<<(std::ostream& out, const OutputDecl& obj)
{
    out << "Val(" << obj.exop.name() << ":" << obj.pos << ")";
    return out;
}

//================================================================================================
// ExOp
//================================================================================================

ExOp::ExOp(ComputationDecl& cgraph, op_ptr _op, bool create_value)
    : ExecutionGraphElt{cgraph.execution_graph}
    , computation_graph{cgraph}
    , tensor_decl{nullptr}
    , tensor_view{nullptr}
    , ref_ops{}
    , op{_op}
    , liveness_live_list{}
    , liveness_free_list{}
    , liveness_new_list{}
    , args{}
    , write_args{}
    , values{}
{
    // moved from ExecuteExOp
    if (op != nullptr)
    {
        computation_graph.ops[op] = this;
        add_ref_op(op);
    }
    // endmoved from ExecuteExOp

    // TODO shim Op needs to be fixed
    // for (input_decl_ptr arg : op->args)
    // {
    //     arg = arg.effective_tensor_op();
    //     exop_ptr exop = computation_graph.get_exop(arg);
    //     output_decl_ptr value = exop->values[0];
    //     add_arg(value);
    // }

    if (create_value && op->is_tensor_op())
    {
        tensor_description_ptr tdesc = op->tensor_description();
        tensor_decl_ptr        tdecl = computation_graph.get_tensor_decl(op);
        add_value(tdecl, tdesc);
    }
}

std::ostream& ngraph::operator<<(std::ostream& out, const ExOp& obj)
{
    out << obj.op->name();
    std::vector<std::string> args;
    for (const InputDecl& id : obj.args)
    {
        std::stringstream ss;
        ss << id.value();
        args.push_back(ss.str());
    }
    out << "\n\targs: " << join(args, ", ");
    out << "\n\tvalues: " << join(obj.values, ", ");
    out << "\n\tlive: " << join(obj.liveness_live_list, ", ");
    out << "\n\tnew: " << join(obj.liveness_new_list, ", ");
    out << "\n\tfree: " << join(obj.liveness_free_list, ", ");
    return out;
}

exop_ptr ExOp::literal_scalar_exop(scalar_t scalar, ComputationDecl& computation_graph)
{
    op_ptr   op                                    = std::make_shared<LiteralScalarOp>(scalar);
    exop_ptr exop                                  = std::make_shared<ExOp>(computation_graph, op);
    exop->values[0].tensor_decl()->is_compile_only = true;
    return exop;
}

InputDecl& ExOp::add_arg(OutputDecl& value, tensor_description_ptr tensor_description)
{
    args.emplace_back(*this, args.size(), tensor_description, &value);
    return args.back();
}

// InputDecl& ExOp::add_write_arg(OutputDecl& value, tensor_description_ptr tensor_description)
// {
//     input_decl_ptr arg = std::make_shared<InputDecl>(this,
//                               args.size(),
//                               value,
//                               tensor_description);
//     write_args.push_back(arg);
//     return write_args.back();
// }

OutputDecl& ExOp::add_value(tensor_decl_ptr tdecl, tensor_description_ptr tensor_description)
{
    if (tensor_description == nullptr)
    {
        tensor_description = tdecl->tensor_description_base;
    }
    values.emplace_back(*this, values.size(), tdecl, tensor_description);
    return values.back();
}

// // void ExOp::take_value(output_decl_ptr value)
// // {
// //     // TODO: this is not going to work ExOpNode is not an ExOp and it is not
// //     // a shared pointer. exop is shared_ptr<ExOp> so we can't get there from
// //     // here
// //     value->exop = this;
// //     value->pos = values.size();
// // }

op_ptr ExOp::get_op()
{
    return op;
}

void ExOp::set_op(op_ptr _op)
{
    if (_op == nullptr)
    {
        if (op == nullptr)
        {
            throw std::invalid_argument("Cannot set op to None.");
        }
        return;
    }
    if (_op->is_tensor_op())
    {
        op_ptr tensor_op = _op->tensor();
        if (_op != tensor_op && tensor_op->is_state_op() == false)
        {
            add_ref_op(_op);
            _op = tensor_op;
        }
    }

    op = _op;
    if (op != nullptr)
    {
        add_ref_op(op);
    }
}

void ExOp::add_ref_op(op_ptr _op)
{
    // Add another op that references this exop.

    // Args:
    //     _op: The computation graph op freferencing this exop.
    ref_ops.push_back(_op);
    computation_graph.ops[_op] = this;
}

size_t ExOp::memory_usage()
{
    // Get the memory usage of this op which is the sum of the sizes of all
    // off the live tensors.

    // Arguments:
    //   None

    // Returns:
    //   Memory usage in bytes
    size_t size = 0;
    for (tensor_decl_ptr node : liveness_live_list)
    {
        size += node->size;
    }
    return size;
}

size_t ExOp::memory_footprint()
{
    // Get the total memory footprint of this op. The footprint hightest memory
    // address used by the tensors in this op

    // Arguments:
    //   None

    // Returns:
    //   Memory footprint in bytes
    size_t max_mem = 0;
    for (tensor_decl_ptr node : liveness_live_list)
    {
        size_t offset = node->size + node->buffer_pool_offset;
        max_mem       = std::max(offset, max_mem);
    }
    return max_mem;
}

size_t ExOp::memory_efficiency()
{
    size_t mem = 100;
    if (memory_footprint() > 0)
    {
        mem = round(float(memory_usage()) / float(memory_footprint()) * 100);
        mem = size_t(mem);
    }
    return mem;
}

bool ExOp::is_exop_end_of_list()
{
    // Returns:
    //     True if this represents the guard past the exop list. See ExOpBlock.
    return false;
}

std::string ExOp::name() const
{
    return op->name();
}

//================================================================================================
// ExOpBlock
//================================================================================================

ExOpBlock::ExOpBlock(ComputationDecl& cgraph)
    : ExecutionGraphElt{cgraph.execution_graph}
    , computation_graph{cgraph}
    , root_set{}
{
}

bool ExOpBlock::is_exop_end_of_list()
{
    // Returns:
    //     True if this represents the guard past the exop list. See ExecuteOp.
    return true;
}

void ExOpBlock::add_ops(std::initializer_list<computation_op_ptr> roots, exop_ptr after_exop)
{
    // Add exops needed to compute ops in roots.

    // Args:
    //     roots: A collection of ops whose values are needed.
    //     after_exop: Where in the list to add the ops. Defaults to the end.
    // Get computation graph ops that have already been computed
    std::vector<op_ptr> computed_ops;
    auto                op_iterator = op_list.end();
    if (after_exop)
    {
        op_iterator = find(op_list.begin(), op_list.end(), after_exop);
    }
    while (op_iterator != op_list.begin())
    {
        exop_ptr exop = *op_iterator--;
        computed_ops.push_back(exop->op);
        computed_ops.insert(computed_ops.end(), exop->ref_ops.begin(), exop->ref_ops.end());
        for (InputDecl& arg : exop->args)
        {
            computed_ops.push_back(arg.exop.op);
        }
        for (InputDecl& arg : exop->args)
        {
            auto ref_ops = arg.value()->exop.ref_ops;
            computed_ops.insert(computed_ops.end(), ref_ops.begin(), ref_ops.end());
        }
    }

    std::vector<op_ptr>                   available;
    std::map<op_ptr, size_t>              counts;
    std::map<op_ptr, std::vector<op_ptr>> parents;
    std::vector<op_ptr>                   ready;

    available.insert(available.end(), roots.begin(), roots.end());
    while (available.size() > 0)
    {
        op_ptr op = available.back();
        available.pop_back();

        if (contains_key(counts, op) || contains(computed_ops, op))
        {
            continue;
        }

        std::vector<op_ptr> children;
        for (op_ptr child : op->all_deps())
        {
            if (!contains(computed_ops, child))
            {
                children.push_back(child);
            }
        }
        // children = OrderedSet((child for child in op.all_deps if child not in computed_ops))
        if (children.size() > 0)
        {
            counts[op] = children.size();
            for (op_ptr child : children)
            {
                parents[child].push_back(op);
                available.push_back(child);
            }
        }
        else
        {
            ready.push_back(op);
        }
    }

    while (ready.size() > 0)
    {
        op_ptr op = ready.back();
        ready.pop_back();
        after_exop = add_op(op, after_exop = after_exop);
        for (op_ptr p : parents[op])
        {
            size_t count = counts[p] - 1;
            if (count == 0)
            {
                ready.push_back(p);
                counts.erase(p);
            }
            else
            {
                counts[p] = count;
            }
        }
    }
    if (counts.size() > 0)
    {
        throw std::runtime_error("Graph not a DAG");
    }
}

exop_ptr ExOpBlock::add_op(op_ptr op, exop_ptr after_exop)
{
    // Add an exop for op to be executed after after_exop.

    // Args:
    //     op: The op.
    //     after_exop: The exop to precede op.

    // Returns:
    //     The new last op. If the op is executable, it will be the added exop,
    //     othwerwise the previous after_exop.
    if (after_exop == nullptr)
    {
        after_exop = op_list.back();
    }
    if (op->is_sequencing_op())
    {
        return after_exop;
    }

    exop_ptr exop = std::make_shared<ExOp>(computation_graph, op);
    return add_exop(exop, after_exop);
}

exop_ptr ExOpBlock::add_exop(exop_ptr exop, exop_ptr after_exop)
{
    // Add exop to the list of exops, after after_exop.

    // Args:
    //     exop:
    //         The exop to add.

    //     after_exop:
    //         If specified, the exop that should be added after after_exop. Defaults to the
    //         last exop added.

    // Returns:
    //     The exop.
    if (after_exop == nullptr)
    {
        op_list.push_back(exop);
    }
    else
    {
        auto it = find(op_list.begin(), op_list.end(), after_exop);
        if (it == op_list.end())
        {
            throw std::runtime_error("exop not found in op_list");
        }
        // list::insert inserts BEFORE the op, we want after so increment iterator
        it++;
        op_list.insert(it, exop);
    }

    return exop;
}

void ExOpBlock::move_exop_to_after_exop(exop_ptr exop, exop_ptr after_exop)
{
    auto it = find(op_list.begin(), op_list.end(), exop);
    op_list.erase(it);
    add_exop(exop, after_exop);
}

void ExOpBlock::remove_exop(exop_ptr exop)
{
    auto it = find(op_list.begin(), op_list.end(), exop);
    op_list.erase(it);
    for (InputDecl& arg : exop->args)
    {
        arg.value()->value_users.erase(&arg);
    }
}

// void ExOpBlock::replace_op(op_ptr old_op, op_ptr new_op)
// {
//     // TODO Replacing an op can remove ops. For example, (x + 2) * 1 -> x + 2
//     // replaces the * with +, so * and 1 drop out
//     // 1 dropping out means one less constant tensor, if it's not used
//     // anywhere else
//     // * dropping out means a change to sequencing.
//     new_op = as_op(new_op)
//     old_exop = computation_graph.get_exop(old_op)
//     new_exop = computation_graph.get_exop(new_op, None)
//     if (new_exop == nullptr)
//     {
//         add_ops([new_op], after_exop=old_exop.prev_exop)
//         new_exop = computation_graph->get_exop(new_op, None)
//     }
//     replace_users(old_exop, new_exop)
//     remove_exop(old_exop)
// }

void ExOpBlock::replace_users(exop_ptr old_exop, exop_ptr new_exop)
{
    // // Replace all users of old_exop with new_exop.

    // // Args:
    // //     old_exop: The original exop.
    // //     new_exop: The replacment exop.
    // for (int i=0; i<old_exop->values.size(); i++)
    // {
    //     OutputDecl* old_value = &old_exop->values[i];
    //     OutputDecl* new_value = &new_exop->values[i];
    //     replace_value(old_value, new_value);
    // }
    // for (op_ptr op : old_exop->ref_ops)
    // {
    //     new_exop->add_ref_op(op);
    // }
    // computation_graph.ops[old_exop->op] = new_exop;
}

// void ExOpBlock::replace_value(OutputDecl* old_value, OutputDecl* new_value)
// {
//     for (InputDecl* value_user : old_value->value_users)
//     {
//         value_user->value(*new_value);
//     }
//     new_value->tensor_decl()->merge_flags(*old_value->tensor_decl());
//     old_value->exop.values[old_value->pos] = *new_value;
// }

void ExOpBlock::replace_exop(exop_ptr old_exop, exop_ptr new_exop)
{
    // add_exop(new_exop, old_exop->prev_exop);
    // This SHOULD be the same as above
    add_exop(new_exop, old_exop);

    replace_users(old_exop, new_exop);
    remove_exop(old_exop);
}

void ExOpBlock::merge_exop(exop_ptr old_exop, exop_ptr new_exop)
{
    // new_exop, which should already exist, takes over for old_exop.

    // Args:
    //     old_exop:
    //     new_exop:
    replace_users(old_exop, new_exop);
    remove_exop(old_exop);
}

size_t ExOpBlock::memory_footprint()
{
    size_t max_mem = 0;
    for (exop_ptr exop : *this)
    {
        max_mem = std::max(exop->memory_footprint(), max_mem);
    }
    return max_mem;
}

size_t ExOpBlock::worst_case_footprint()
{
    size_t mem = 0;
    for (OutputDecl* value : get_temp_vars())
    {
        mem += value->write_view()->tensor_decl.size;
    }
    return mem;
}

size_t ExOpBlock::memory_efficiency()
{
    size_t footprint = memory_footprint();
    size_t usage     = 0;
    for (exop_ptr exop : op_list)
    {
        usage = std::max(usage, exop->memory_usage());
    }
    size_t result = 100;
    if (footprint > 0)
    {
        result = int(round((float(usage) / float(footprint)) * 100));
    }
    return result;
}

size_t ExOpBlock::persistent_size()
{
    size_t mem = 0;
    for (OutputDecl* value : get_persistent_vars())
    {
        mem += value->write_view()->tensor_decl.size;
    }
    return mem;
}

std::set<OutputDecl*> ExOpBlock::get_vars()
{
    std::set<OutputDecl*> vars;
    for (exop_ptr exop : op_list)
    {
        for (InputDecl& value : exop->args)
        {
            vars.insert(value.value());
        }
        for (OutputDecl& value : exop->values)
        {
            vars.insert(&value);
        }
    }
    return vars;
}

std::set<OutputDecl*> ExOpBlock::get_temp_vars()
{
    std::set<OutputDecl*> result;
    for (OutputDecl* value : get_vars())
    {
        if (value->write_view()->tensor_decl.is_persistent == false)
        {
            result.insert(value);
        }
    }
    return result;
}

std::set<OutputDecl*> ExOpBlock::get_persistent_vars()
{
    std::set<OutputDecl*> result;
    for (OutputDecl* value : get_vars())
    {
        if (value->write_view()->tensor_decl.is_persistent)
        {
            result.insert(value);
        }
    }
    return result;
}

//================================================================================================
// TensorDecl
//================================================================================================

TensorDecl::TensorDecl(ExecutionGraph&        eg,
                       ElementType            _element_type,
                       size_t                 _size,
                       bool                   _is_persistent,
                       bool                   _is_input,
                       tensor_description_ptr _tensor_description_base,
                       bool                   _is_output,
                       bool                   _is_constant,
                       tensor_description_ptr tensor_description,
                       bool                   _is_compile_only)
    : ExecutionGraphElt{eg}
    , element_type{_element_type}
    , size{_size}
    , is_persistent{_is_persistent}
    , is_input{_is_input}
    , is_output{_is_output}
    , buffer_pool_offset{0}
    , tensor_view_decls{}
    , tensor_description_base{_tensor_description_base}
    , lifespan{0}
    , is_constant{_is_constant}
    , is_compile_only{_is_compile_only}
    , initial_value{nullptr}
    , source_tensor{this}
{
    // TODO: fix this somehow
    // if (tensor_description == nullptr)
    // {
    //     if (op == nullptr)
    //     {
    //         tensor_description = tensor_description_base;
    //     }
    //     else
    //     {
    //         if (op->tensor()->is_state_op())
    //         {
    //             initial_value = op->tensor()->initial_value;
    //         }
    //         tensor_description = op->tensor_description();
    //     }
    // }

    // // TODO Needed for initialization. Use exop value instead.
    // add_value(this, tensor_description)
}

tensor_view_decl_ptr
    TensorDecl::get_tensor_view(tensor_description_ptr tdesc, InputDecl* reader, OutputDecl* writer)
{
    tensor_view_decl_ptr tensor_view;
    if (tdesc == nullptr)
    {
        tdesc = tensor_description_base;
    }
    tensor_view = tensor_view_decls[tdesc->axes_key()];
    if (tensor_view == nullptr)
    {
        tensor_view = std::make_shared<TensorViewDecl>(*this, tdesc, execution_graph);
        tensor_view_decls[tdesc->axes_key()] = tensor_view;
    }
    if (reader == nullptr)
    {
        tensor_view->readers.insert(reader);
    }
    if (writer != nullptr)
    {
        tensor_view->writers.insert(writer);
    }
    return tensor_view;
}

tensor_view_decl_ptr TensorDecl::get_tensor_view(tensor_description_ptr tdesc, InputDecl* reader)
{
    return get_tensor_view(tdesc, reader, nullptr);
}

tensor_view_decl_ptr TensorDecl::get_tensor_view(tensor_description_ptr tdesc, OutputDecl* writer)
{
    return get_tensor_view(tdesc, nullptr, writer);
}

void TensorDecl::merge_flags(const TensorDecl& tensor)
{
    is_input |= tensor.is_input;
    is_persistent |= tensor.is_persistent;
    is_output |= tensor.is_output;
}

tensor_description_ptr TensorDecl::buffer_key()
{
    // Returns: The key that makes this tensor unique in a buffer.
    return tensor_description_base;
}

std::string TensorDecl::prefix()
{
    std::stringstream ss{"_a"};
    ss << "a_";
    if (!is_persistent)
    {
        ss << execution_graph.computation_decl->computation_op->name();
    }
    return ss.str();
}

std::string TensorDecl::variable_name()
{
    std::stringstream ss;
    ss << prefix() << "_" << tensor_name();
    return ss.str();
}

std::string TensorDecl::tensor_name()
{
    // Returns: Name used for the tensor.
    return tensor_description_base->name();
}

std::string TensorDecl::buffer_name()
{
    // Returns: Name used for the buffer.
    return tensor_description_base->name();
}

// std::string TensorDecl::name()
// {
//     return op->name();
// }

std::ostream& ngraph::operator<<(std::ostream& out, const TensorDecl& obj)
{
    out << obj.tensor_description_base->name();
    return out;
}

//================================================================================================
// TensorViewDecl
//================================================================================================

TensorViewDecl::TensorViewDecl(TensorDecl&            _tensor_decl,
                               tensor_description_ptr _tensor_description,
                               ExecutionGraph&        eg)
    : ExecutionGraphElt{eg}
    , tensor_decl{_tensor_decl}
    , tensor_description{_tensor_description}
    , readers{}
    , writers{}
    , value{nullptr}
{
    // self.value = None
}

std::string TensorViewDecl::name() const
{
    std::stringstream ss;
    ss << tensor_decl.variable_name() << "_v_" << tensor_description->name();
    ss << "_" << join(tensor_description->shape(), "x");
    return ss.str();
    // shape_str = "x".join((str(_) for _ in tensor_description.shape))
    // return "{}_v_{}_{}".format(self.tensor_decl.variable_name,
    //                            self.tensor_description.name,
    //                            shape_str)
}

// op_ptr TensorViewDecl::op()
// {
//     return tensor_decl->op;
// }

tensor_view_decl_ptr TensorViewDecl::get_tensor_view(tensor_description_ptr _tensor_description,
                                                     InputDecl*             _reader,
                                                     OutputDecl*            _writer)
{
    return tensor_decl.get_tensor_view(_tensor_description, _reader, _writer);
}

tensor_view_decl_ptr TensorViewDecl::get_tensor_view(tensor_description_ptr _tensor_description,
                                                     InputDecl*             _reader)
{
    return tensor_decl.get_tensor_view(_tensor_description, _reader, nullptr);
}

tensor_view_decl_ptr TensorViewDecl::get_tensor_view(tensor_description_ptr _tensor_description,
                                                     OutputDecl*            _writer)
{
    return tensor_decl.get_tensor_view(_tensor_description, nullptr, _writer);
}

//================================================================================================
// ComputationDecl
//================================================================================================

ComputationDecl::ComputationDecl(ExecutionGraph& eg, computation_op_ptr op)
    : ExecutionGraphElt{eg}
    , computation_op{op}
{
    exop_block = std::make_shared<ExOpBlock>(*this);
    exop_block->add_ops({computation_op});

    // returns = std::make_shared<ReturnExOp>(*this);
    auto return_op = std::make_shared<ReturnOp>();
    returns        = std::make_shared<ExOp>(*this, return_op);

    // Get the exops we need values for, so that if they are computed at compile-time we still
    // have a view to their value.
    for (op_ptr co : computation_op->values())
    {
        if (co->is_tensor_op())
        {
            exop_block->root_set.insert(get_exop(co));
        }
    }
    for (op_ptr co : computation_op->values())
    {
        if (co->is_tensor_op())
        {
            exop_block->root_set.insert(get_exop(op));
        }
    }

    for (ExOp* e : exop_block->root_set)
    {
        for (OutputDecl& value : e->values)
        {
            InputDecl& arg                            = returns->add_arg(value);
            op_returns[e->op.get()]                   = &arg;
            op_returns[e->op->tensor().get()]         = &arg;
            value.write_view()->tensor_decl.is_output = true;
        }
    }

    for (op_ptr co : computation_op->values())
    {
        if (co->tensor()->is_tensor_op())
        {
            values.insert(get_exop(op));
        }
    }
}

tensor_decl_ptr ComputationDecl::get_tensor_decl(op_ptr _op)
{
    return execution_graph.get_tensor_decl(_op);
}

ExOp* ComputationDecl::get_exop(op_ptr _op)
{
    op_ptr original_op = _op;
    _op                = _op->effective_tensor_op();
    if (_op->is_state_op())
    {
        throw std::runtime_error("Use get_tensor for AssignableTensorOp");
    }
    ExOp* exop = ops[_op];
    if (exop != nullptr)
    {
        return exop;
    }
    // if (default_value != _default_default)
    // {
    //     return default_value;
    // }
    std::stringstream ss;
    ss << "Unhandled op: " << original_op;
    throw std::runtime_error(ss.str());
}

//================================================================================================
// ExecutionState
//================================================================================================

ExecutionState::ExecutionState(transformer_ptr transformer)
    : __transformer{transformer}
    , __tensors_decls{}
{
}

transformer_ptr ExecutionState::transformer()
{
    return __transformer;
}

execution_graph_ptr ExecutionState::make_execution_graph(computation_op_ptr computation_op)
{
    return std::make_shared<ExecutionGraph>(*this, computation_op);
}

tensor_decl_ptr ExecutionState::get_op_tensor(op_ptr op)
{
    tensor_description_ptr tensor_description      = op->tensor_description();
    tensor_description_ptr tensor_description_base = tensor_description->base();
    return __tensors_decls[tensor_description_base];
}

tensor_decl_ptr ExecutionState::ensure_tensor_decl(ExecutionGraph&        execution_graph,
                                                   tensor_description_ptr tensor_description,
                                                   op_ptr                 op)
{
    tensor_description_ptr tensor_description_base = tensor_description->base();
    tensor_decl_ptr        tensor_decl             = __tensors_decls[tensor_description_base];
    if (tensor_decl == nullptr)
    {
        bool is_output       = false;
        bool is_constant     = false;
        bool is_compile_only = false;

        tensor_decl                              = std::make_shared<TensorDecl>(execution_graph,
                                                   tensor_description_base->element_type(),
                                                   tensor_description_base->tensor_size(),
                                                   tensor_description_base->is_persistent(),
                                                   tensor_description_base->is_input(),
                                                   tensor_description_base,
                                                   is_output,
                                                   is_constant,
                                                   nullptr,
                                                   is_compile_only);
        __tensors_decls[tensor_description_base] = tensor_decl;
    }
    return tensor_decl;
}

//================================================================================================
// ExecutionGraph
//================================================================================================

ExecutionGraph::ExecutionGraph(ExecutionState& es, computation_op_ptr computation_op)
    : execution_state{es}
    , tensor_decls{}
    , computation_decl{std::make_shared<ComputationDecl>(*this, computation_op)}
{
}

tensor_decl_ptr ExecutionGraph::get_tensor_decl(op_ptr                 op,
                                                tensor_description_ptr tensor_description)
{
    if (tensor_description == nullptr)
    {
        tensor_description = op->tensor_description();
    }
    tensor_description_ptr tensor_description_base = tensor_description->base();
    if (tensor_description_base->is_persistent())
    {
        return execution_state.ensure_tensor_decl(*this, tensor_description, op);
    }
    tensor_decl_ptr tensor_decl = tensor_decls[tensor_description_base];
    if (tensor_decl == nullptr)
    {
        bool is_output       = false;
        bool is_constant     = false;
        bool is_compile_only = false;

        tensor_decl                           = std::make_shared<TensorDecl>(*this,
                                                   tensor_description_base->element_type(),
                                                   tensor_description_base->tensor_size(),
                                                   tensor_description_base->is_persistent(),
                                                   tensor_description_base->is_input(),
                                                   tensor_description_base,
                                                   is_output,
                                                   is_constant,
                                                   nullptr,
                                                   is_compile_only);
        tensor_decls[tensor_description_base] = tensor_decl;
    }
    return tensor_decl;
}
