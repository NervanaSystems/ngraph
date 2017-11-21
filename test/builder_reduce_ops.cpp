#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

TEST(BUILDER_REDUCE, SUM) 
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(builder::Sum(A, {0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{9, 12}), result->get_vector<float>());
}