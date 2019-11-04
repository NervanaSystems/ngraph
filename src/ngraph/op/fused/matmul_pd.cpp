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
#include <memory>
#include <numeric>

#include "matmul_pd.hpp"
#include "ngraph/builder/matmul_factory.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::MatMulPd::type_info;
op::MatMulPd::MatMulPd(const Output<Node>& A,
                   const Output<Node>& B,
                   const bool& transpose_a,
                   const bool& transpose_b)
    : FusedOp(OutputVector{A, B})
    , m_transpose_a{transpose_a}
    , m_transpose_b{transpose_b}
{
    constructor_validate_and_infer_types();
}

template<class Input>
void DecomposeLogic(Input& input, bool transpose, bool reverse=false)
{
    std::cout << ">> MatMulPDForward << " << std::endl; 
    auto _rank = input.get_shape().size();
    if(_rank < 2)
    {
        if(_rank)
        {            
            if(reverse)
            {
               input = make_shared<op::Reshape>(input, AxisVector{0},Shape{input.get_shape()[0],1});
            }
            else
            {
               input = make_shared<op::Reshape>(input, AxisVector{0},Shape{1,input.get_shape()[0]});
            }
        }
        else
        {
            input = make_shared<op::Reshape>(input,AxisVector{},Shape{1, 1});
        }              
    _rank=2;     
    }
    if (transpose)
    {
        vector<size_t> _axes_order(_rank);
        iota(_axes_order.begin(), _axes_order.end(), 0);
        swap(_axes_order[_rank - 1], _axes_order[_rank - 2]);
        input = builder::reorder_axes(input, _axes_order);
    }
}

inline NodeVector remove_1(std::shared_ptr<ngraph::Node> input_node)
{
    auto _input_shape = input_node->get_shape();
    AxisVector _axis( _input_shape.size() );
    iota(_axis.begin(),_axis.end(),0);
    Shape _shape(_input_shape.begin(),_input_shape.end());
    auto _b_remove = std::remove(_shape.begin(),_shape.end(),1);
    _shape.erase(_b_remove,_shape.end());
    Output<Node> _node( input_node );  
    auto _reshape = make_shared<op::Reshape>( _node , _axis, _shape);
    NodeVector _final_vector{ _reshape };
    return _final_vector; 
}


NodeVector op::MatMulPd::decompose_op() const
{
     auto _A = input_value(0);
     auto _B = input_value(1);
     DecomposeLogic(_A,m_transpose_a);
     DecomposeLogic(_B,m_transpose_b,true);
     builder::MatmulFactory factory({_A, _B});
     auto _node_vector_matmul = factory.make_matmul_op();
     if(_node_vector_matmul.size()!=1)
     {          
         // throw error ?
     }
     auto _first_item_node_vector = _node_vector_matmul[0]; 
     if(!_first_item_node_vector)
     {
         // throw error  
     }
     auto _b = _first_item_node_vector->get_shape().begin();
     auto _e = _first_item_node_vector->get_shape().end();  
     auto _it = std::find(_b,_e,1);
     if( _it != _e)
     {
         _node_vector_matmul = remove_1(_first_item_node_vector);
     }   
          
     return _node_vector_matmul;
}


shared_ptr<Node> op::MatMulPd::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMulPd>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}
