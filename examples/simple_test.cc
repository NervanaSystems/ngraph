// compile and test as follows.
// g++ -std=c++11 simple_test.cc -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -lngraph -lpthread -lgtest -o /tmp/test
// env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib /tmp/test
#include <bits/stdc++.h>
#include <ngraph/ngraph.hpp>
#include "gtest/gtest.h"
using namespace std;
using namespace ngraph;
template <typename T>
vector<T> read_vector(shared_ptr<ngraph::runtime::TensorView> tv) {
  if (ngraph::element::from<T>() !=
      tv->get_tensor_view_layout()->get_element_type()) {
    throw invalid_argument("read_vector type must match TensorView type");
  }
  size_t element_count = ngraph::shape_size(tv->get_shape());
  size_t size = element_count * sizeof(T);
  vector<T> rc(element_count);
  tv->read(rc.data(), 0, size);
  return rc;
}
template <typename T>
void copy_data(shared_ptr<ngraph::runtime::TensorView> tv,
               const vector<T>& data) {
  size_t data_size = data.size() * sizeof(T);
  tv->write(data.data(), 0, data_size);
}

TEST(simple, test) {
  auto manager = runtime::Manager::get("INTERPRETER");
  auto backend = manager->allocate_backend();

  auto shape = Shape{2, 2};
  auto X = make_shared<op::Parameter>(element::f32, shape);
  auto Y = make_shared<op::Parameter>(element::f32, shape);

  auto op = make_shared<op::Divide>(X, Y);

  auto f = make_shared<Function>(op, vector<shared_ptr<op::Parameter>>{X, Y});

  auto C = make_shared<op::Parameter>(element::f32, shape);
  vector<shared_ptr<Node>> dYdXs;
  for (auto param : {X, Y}) {
    dYdXs.push_back(op->backprop_node(param, C));
  }

  auto bf =
      make_shared<Function>(dYdXs, vector<shared_ptr<op::Parameter>>{C, X, Y});

  auto forward_external = manager->compile(f);
  auto f_cf = backend->make_call_frame(forward_external);

  auto backward_external = manager->compile(bf);
  auto bf_cf = backend->make_call_frame(backward_external);

  auto a = backend->make_primary_tensor_view(element::f32, shape);
  copy_data(a, vector<float>{2, 4, 8, 16});
  auto b = backend->make_primary_tensor_view(element::f32, shape);
  copy_data(b, vector<float>{1, 2, 4, 8});
  auto result = backend->make_primary_tensor_view(element::f32, shape);

  f_cf->call({a, b}, {result});

  EXPECT_EQ((vector<float>{2, 2, 2, 2}), read_vector<float>(result));

  auto c = backend->make_primary_tensor_view(element::f32, shape);
  copy_data(c, vector<float>{1, 1, 1, 1});

  auto da = backend->make_primary_tensor_view(element::f32, shape);
  auto db = backend->make_primary_tensor_view(element::f32, shape);

  bf_cf->call({c, a, b}, {da, db});

  EXPECT_EQ((vector<float>{1, 0.5, 0.25, 0.125}), read_vector<float>(da));
  EXPECT_EQ((vector<float>{-2, -1, -0.5, -0.25}), read_vector<float>(db));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
