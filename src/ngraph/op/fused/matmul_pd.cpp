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


#include "ngraph/op/dot.hpp"
#include "ngraph/op/experimental/batch_mat_mul.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/sum.hpp"

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


// #############################
constexpr NodeTypeInfo op::MatMulPdBackward::type_info;
  op::MatMulPdBackward::MatMulPdBackward(
      std::shared_ptr<ngraph::Node> A,
      std::shared_ptr<ngraph::Node> B,
      std::shared_ptr<ngraph::Node> OutGrad,
      bool is_X, 
      bool is_Y,
      bool transpose_a, 
      bool transpose_b ) : FusedOp(OutputVector{A, B, OutGrad}),
      m_A{A}, m_B{B}, m_is_X{is_X}, m_is_Y(is_Y),
      m_transpose_a{transpose_a}, m_transpose_b{transpose_b}
      {
         constructor_validate_and_infer_types();
      }

      std::shared_ptr<ngraph::Node> op::MatMulPdBackward::helper_dotOp(
          const std::shared_ptr<ngraph::Node>& a,
          const std::shared_ptr<ngraph::Node>& b) const
          {
           std::shared_ptr<ngraph::Node> out;  
            auto a_shape = a->get_shape();
            auto na = a_shape.size();
            auto b_shape = b->get_shape();
            auto nb = b_shape.size();
            if (na > 2 && nb > 2) {
               out = std::make_shared<op::BatchMatMul>(a, b);
            } else {
               out = std::make_shared<op::Dot>(a, b);
            }
            return out;
         } 

       std::shared_ptr<ngraph::Node> op::MatMulPdBackward::helper_reshapeToOriginal(
       std::shared_ptr<ngraph::Node> input, const ngraph::Shape& shape) const
       {
       auto input_shape = input->get_shape();
       std::vector<size_t> axis(input_shape.size());
       std::iota(axis.begin(), axis.end(), 0);
       auto out = std::make_shared<ngraph::op::Reshape>(input, axis, shape);
       return out;
       }


std::shared_ptr<ngraph::Node> op::MatMulPdBackward::helper_transposeAndFlat3D (
    const std::shared_ptr<ngraph::Node>& input, const bool transpose,
    bool x) const {
    auto shape = input->get_shape();
    size_t n = shape.size();
     std::shared_ptr<ngraph::Node> output;
    if (n >= 3) {
    std::vector<size_t> order(n);
    std::iota(std::begin(order), std::end(order), 0);
    size_t outer = 1;
    for (size_t i = 0; i < n - 2; i++) {
      outer = outer * shape[i];
    }
    std::vector<size_t> reshape{outer, shape[n - 2], shape[n - 1]};

    if (transpose == true) {
      order[n - 2] = n - 1;
      order[n - 1] = n - 2;
      reshape[2] = shape[n - 2];
      reshape[1] = shape[n - 1];
   //   auto reshape_order = std::make_shared<ngraph::op::Constant>(order, ngraph::Shape{order.size()});
    }
    output = std::make_shared<ngraph::op::Reshape>(
        input, ngraph::AxisVector(order), ngraph::Shape(reshape));
  } else {
    std::shared_ptr<ngraph::Node> temp;
    if (n == 1 && x == true) {
      temp = std::make_shared<ngraph::op::Reshape>(input, ngraph::AxisVector{0},
                                                   ngraph::Shape{1, shape[0]});
    } else if (n == 1 && x == false) {
      temp = std::make_shared<ngraph::op::Reshape>(input, ngraph::AxisVector{0},
                                                   ngraph::Shape{shape[0], 1});
    } else {
      temp = input;
    }
    auto temp_shape = temp->get_shape();
    if (transpose == true) {
      output = std::make_shared<ngraph::op::Reshape>(
          temp, ngraph::AxisVector{1, 0},
          ngraph::Shape{temp_shape[1], temp_shape[0]});
    } else {
      output = temp;
    }
  }
  return output;
}


std::shared_ptr<ngraph::Node> op::MatMulPdBackward::helper_broadcast3D(
    const std::shared_ptr<ngraph::Node>& input, size_t axis0) const {
  auto shape = input->get_shape();
  size_t n = shape.size();
  if (n == 2) {
    auto output = std::make_shared<ngraph::op::Broadcast>(
        input, ngraph::Shape{axis0, shape[0], shape[1]}, ngraph::AxisSet{0});
    return output;
  }
  return input;
}


      NodeVector op::MatMulPdBackward::decompose_op() const 
      {
            auto x = input_value(0).get_node_shared_ptr();
            auto y = input_value(1).get_node_shared_ptr();
            auto dout = input_value(2).get_node_shared_ptr();
     
        //  auto& dout = OutGrad;
        //  auto& x = m_A;
        //  auto& y = m_B;
         auto dout_shape = dout->get_shape();
         auto x_shape = x->get_shape();
         auto y_shape = y->get_shape();
         size_t nx = x_shape.size();
         size_t ny = y_shape.size();
         size_t ndout = dout_shape.size();
         std::shared_ptr<ngraph::Node> x2, y2;
         std::shared_ptr<ngraph::Node> dout2;

  x2 = helper_transposeAndFlat3D(x, false);
  y2 = helper_transposeAndFlat3D(y, false, false);
  dout2 = helper_transposeAndFlat3D(dout, false);
  auto x2_shape = x2->get_shape();
  auto y2_shape = y2->get_shape();
  if (nx >= 3 || ny >= 3) {
    std::shared_ptr<ngraph::Node> dout_temp;
    if (ndout == 2) {
      dout_temp = std::make_shared<ngraph::op::Reshape>(
          dout, ngraph::AxisVector{0, 1},
          ngraph::Shape{dout_shape[0], dout_shape[1], 1});
      if (ny < 3) {
        dout2 = dout_temp;
      } else {
        dout2 = helper_transposeAndFlat3D(dout_temp, true);
      }
    }
    x2 = helper_broadcast3D(x2, y_shape[0]);
    y2 = helper_broadcast3D(y2, x_shape[0]);

  } else {
    dout2 = helper_transposeAndFlat3D(dout, false, nx == 1 && m_transpose_a == false);
  }

  if (m_transpose_b == false) {
    y2 = helper_transposeAndFlat3D(y2, true);
  }
  if (m_transpose_a == false) {
    x2 = helper_transposeAndFlat3D(x2, true);
  }
  auto dx = helper_dotOp(dout2, y2);
  auto dy = helper_dotOp(x2, dout2);
  if (m_transpose_a == true) {
    dx = helper_transposeAndFlat3D(dx, true);
  }
  if (m_transpose_b == true) {
    dy = helper_transposeAndFlat3D(dy, true);
  }

  if (nx < 3 && ny >= 3) {
    dx = std::make_shared<ngraph::op::Sum>(dx, ngraph::AxisSet{0});
  }
  if (ny < 3 && nx >= 3) {
    dy = std::make_shared<ngraph::op::Sum>(dy, ngraph::AxisSet{0});
  }
  
    auto dx_t = helper_reshapeToOriginal(dx, x_shape);
    auto dy_t = helper_reshapeToOriginal(dy, y_shape);

    return NodeVector{ dx_t, dy_t, dout };

    }

/* Not sure ??? */
shared_ptr<Node> op::MatMulPdBackward::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMulPdBackward>(new_args.at(0), new_args.at(1), new_args.at(2), m_is_X, m_is_Y, m_transpose_a, m_transpose_b);
}
