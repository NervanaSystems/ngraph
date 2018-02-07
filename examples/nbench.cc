// compile and run with
// g++ ./nbench.cc -std=c++11 -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -lngraph -o nbench
// env NGRAPH_CPU_EMIT_TIMING=1 ./nbench
#include <bits/stdc++.h>
#include <ngraph/file_util.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/serializer.hpp>
#include "clipp.h"
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
void write_vector(std::shared_ptr<ngraph::runtime::TensorView> tv,
                  const std::vector<T>& values) {
  tv->write(values.data(), 0, values.size() * sizeof(T));
}
template <typename T>
void copy_data(shared_ptr<ngraph::runtime::TensorView> tv,
               const vector<T>& data) {
  size_t data_size = data.size() * sizeof(T);
  tv->write(data.data(), 0, data_size);
}
static multimap<size_t, string> agregate_timing(
    const vector<runtime::PerformanceCounter>& perf_data) {
  unordered_map<string, size_t> timing;
  for (const runtime::PerformanceCounter& p : perf_data) {
    string op = p.name().substr(0, p.name().find('_'));
    timing[op] += p.microseconds();
  }

  multimap<size_t, string> rc;
  for (const pair<string, size_t>& t : timing) {
    rc.insert({t.second, t.first});
  }
  return rc;
}
template <typename T>
class Uniform {
 public:
  Uniform(T min, T max, T seed = 0)
      : m_engine(seed),
        m_distribution(min, max),
        m_r(std::bind(m_distribution, m_engine)) {}

  const std::shared_ptr<runtime::TensorView> initialize(
      const std::shared_ptr<runtime::TensorView>& ptv) {
    std::vector<T> vec = read_vector<T>(ptv);
    for (T& elt : vec) {
      elt = m_r();
    }
    write_vector(ptv, vec);
    return ptv;
  }

 protected:
  std::default_random_engine m_engine;
  std::uniform_real_distribution<T> m_distribution;
  std::function<T()> m_r;
};
void run_benchmark(const string& json_path, const string& backend_name,
                   size_t iterations) {
  string env_var_name = "NGRAPH_" + backend_name + "_EMIT_TIMING";
  bool emit_timing = (std::getenv(env_var_name.c_str()) != nullptr);
  if (!emit_timing) {
    cout << "To get per-op timing set the environment variable " << env_var_name
         << "\n";
  }

  Uniform<float> rng{-1, 1, 0};
  const string json_string = ngraph::file_util::read_file_to_string(json_path);
  stringstream ss(json_string);
  shared_ptr<Function> f = ngraph::deserialize(ss);

  stopwatch build_time;
  build_time.start();
  auto manager = runtime::Manager::get(backend_name);
  auto external = manager->compile(f);
  auto backend = manager->allocate_backend();
  auto cf = backend->make_call_frame(external);
  build_time.stop();
  cout << "build_time " << build_time.get_milliseconds() << "ms" << endl;

  vector<shared_ptr<runtime::TensorView>> args;
  for (shared_ptr<op::Parameter> param : f->get_parameters()) {
    auto tensor = backend->make_primary_tensor_view(param->get_element_type(),
                                                    param->get_shape());
    rng.initialize(tensor);
    args.push_back(tensor);
  }
  vector<shared_ptr<runtime::TensorView>> results;
  for (shared_ptr<Node> out : f->get_results()) {
    auto result = backend->make_primary_tensor_view(out->get_element_type(),
                                                    out->get_shape());
    results.push_back(result);
  }

  stopwatch t1;
  t1.start();
  for (size_t i = 0; i < static_cast<size_t>(iterations); i++) {
    cf->tensor_call(args, results);
  }
  t1.stop();
  float time = t1.get_milliseconds();
  cout << time / iterations << "ms per iteration" << endl;

  vector<runtime::PerformanceCounter> perf_data = cf->get_performance_data();
  sort(perf_data.begin(), perf_data.end(),
       [](const runtime::PerformanceCounter& p1,
          const runtime::PerformanceCounter& p2) {
         return p1.total_microseconds() > p2.total_microseconds();
       });
  multimap<size_t, string> timing = agregate_timing(perf_data);
  for (auto it = timing.rbegin(); it != timing.rend(); it++) {
    cout.imbue(locale(""));
    cout << setw(15) << left << it->second << " " << setw(10) << right
         << it->first << "us\n";
  }
}
int main(int argc, char** argv) {
  string model = "model.json";
  string backend = "CPU";
  int iter = 10;
  auto cli =
      ("model json file to use (default: model.json)" % clipp::option("-f") &
           clipp::value("filename", model),
       "Backed to use (default: CPU)" % clipp::option("-b") &
           clipp::value("backend", backend),
       "Iterations (default: 10)" % clipp::option("-i") &
           clipp::value("iterations", iter));
  if (!clipp::parse(argc, argv, cli) || !static_cast<bool>(ifstream(model))) {
    cout << clipp::make_man_page(cli, argv[0])
                .prepend_section(
                    "DESCRIPTION",
                    "    Benchmark ngraph json model with given backend.");
    return 1;
  }
  cout << "Benchmarking " << model << ", " << backend << " backend, " << iter
       << " iterations.\n";
  run_benchmark(model, backend, iter);
}
