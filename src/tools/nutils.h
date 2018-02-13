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

#include <ngraph/file_util.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/serializer.hpp>
namespace ngraph
{
    using CFrame = std::shared_ptr<runtime::CallFrame>;
    using TViews = std::vector<std::shared_ptr<runtime::TensorView>>;
    using CallFrameIO = std::tuple<CFrame, CFrame, TViews, TViews>;

    CallFrameIO
        get_cfio(std::string backend_type, std::shared_ptr<Function> f, bool backward = false)
    {
        auto manager = runtime::Manager::get(backend_type);
        auto external = manager->compile(f);
        auto backend = manager->allocate_backend();
        auto cf = backend->make_call_frame(external);
        auto result = backend->make_primary_tensor_view(f->get_output_element_type(0),
                                                        f->get_output_shape(0));
        std::vector<std::shared_ptr<runtime::TensorView>> viv;
        for (const auto& i : f->get_parameters())
            viv.push_back(backend->make_primary_tensor_view(i->get_element_type(), i->get_shape()));
        std::vector<std::shared_ptr<runtime::TensorView>> vrv;
        for (int i = 0; i < f->get_output_size(); ++i)
            vrv.push_back(backend->make_primary_tensor_view(f->get_output_element_type(i),
                                                            f->get_output_shape(i)));
        if (!backward)
            return CallFrameIO{cf, nullptr, viv, vrv};
        auto C =
            std::make_shared<op::Parameter>(f->get_output_element_type(0), f->get_output_shape(0));
        std::vector<std::shared_ptr<op::Parameter>> backparam;
        backparam.push_back(C);
        viv.push_back(backend->make_primary_tensor_view(C->get_element_type(), C->get_shape()));
        for (const auto& i : f->get_parameters())
            vrv.push_back(backend->make_primary_tensor_view(i->get_element_type(), i->get_shape()));
        std::vector<std::shared_ptr<Node>> dYdXs;
        auto op = f->get_result();
        for (const auto& i : f->get_parameters())
        {
            dYdXs.push_back(op->backprop_node(i, C));
            backparam.push_back(i);
        }
        auto bf = std::make_shared<Function>(dYdXs, backparam);
        auto backward_external = manager->compile(bf);
        auto bf_cf = backend->make_call_frame(backward_external);
        return CallFrameIO{cf, bf_cf, viv, vrv};
    }

    template <typename T>
    static std::vector<T> read_vector(std::shared_ptr<ngraph::runtime::TensorView> tv)
    {
        if (ngraph::element::from<T>() != tv->get_tensor_view_layout()->get_element_type())
        {
            throw std::invalid_argument("read_vector type must match TensorView type");
        }
        size_t element_count = ngraph::shape_size(tv->get_shape());
        size_t size = element_count * sizeof(T);
        std::vector<T> rc(element_count);
        tv->read(rc.data(), 0, size);
        return rc;
    }
    template <typename T>
    inline void write_vector(std::shared_ptr<ngraph::runtime::TensorView> tv,
                             const std::vector<T>& values)
    {
        tv->write(values.data(), 0, values.size() * sizeof(T));
    }
    template <typename T>
    inline void copy_data(std::shared_ptr<ngraph::runtime::TensorView> tv,
                          const std::vector<T>& data)
    {
        size_t data_size = data.size() * sizeof(T);
        tv->write(data.data(), 0, data_size);
    }
    inline std::multimap<size_t, std::string>
        agregate_timing(const std::vector<runtime::PerformanceCounter>& perf_data)
    {
        std::unordered_map<std::string, size_t> timing;
        for (const runtime::PerformanceCounter& p : perf_data)
        {
            std::string op = p.name().substr(0, p.name().find('_'));
            timing[op] += p.microseconds();
        }

        std::multimap<size_t, std::string> rc;
        for (const std::pair<std::string, size_t>& t : timing)
        {
            rc.insert({t.second, t.first});
        }
        return rc;
    }
    template <typename T>
    class Uniform
    {
    public:
        Uniform(T min, T max, T seed = 0)
            : m_engine(seed)
            , m_distribution(min, max)
            , m_r(std::bind(m_distribution, m_engine))
        {
        }

        const std::shared_ptr<runtime::TensorView>
            initialize(const std::shared_ptr<runtime::TensorView>& ptv)
        {
            std::vector<T> vec = read_vector<T>(ptv);
            for (T& elt : vec)
            {
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
}
