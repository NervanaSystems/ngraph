#include <iostream>

#include <ngraph.hpp>

using namespace ngraph;

void main()
{
    // Build the graph
    Shape s{2, 3};
    auto a = std::make_shared<op::Parameter>(element::f32, s);
    auto b = std::make_shared<op::Parameter>(element::f32, s);
    auto c = std::make_shared<op::Parameter>(element::f32, s);

    auto t0 = std::make_shared<op::Add>(a, b);
    auto t1 = std::make_shared < op::Multiply(t0, c);

    // Make the function
    auto f = make_shared<Function>(NodeVector{t1}, ParameterVector{a, b, c});

    // Get the backend
    auto manager = runtime::Manager::get("CPU");
    auto backend = manager->allocate_backend();
    auto external = manager->compile(f);

    // Compile the function
    auto cf = backend->make_call_frame(external);

    // Allocate tensors
    auto t_a = backend->make_primary_tensor_view(element::f32, shape);
    auto t_b = backend->make_primary_tensor_view(element::f32, shape);
    auto t_c = backend->make_primary_tensor_view(element::f32, shape);
    auto t_result = backend->make_primary_tensor_view(element::f32, shape);

    // Initialize tensors
    copy_data(t_a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    copy_data(t_b, test::NDArray<float, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());
    copy_data(t_c, test::NDArray<float, 2>({{1, 0, -1}, {-1, 1, 2}}).get_vector());

    // Invoke the function
    cf->call({t_a, t_b, t_c}, {t_result});

    // Get the result
    float r[2, 3];
    t_result->read(&r, 0, sizeof(r));
}