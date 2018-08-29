/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <cassert>
#include <deque>
#include <forward_list>
#include <iomanip>
#include <map>
#include <numeric>
#include <unordered_set>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/result_vector.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"

#include <iostream>

using namespace std;
using namespace ngraph;

std::string ngraph::to_cplusplus_sourcecode_literal(bool val)
{
    return val ? "true" : "false";
}

void ngraph::dump(ostream& out, const void* _data, size_t _size)
{
    auto flags = out.flags();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(_data);
    size_t len = _size;
    size_t index = 0;
    while (index < len)
    {
        out << std::hex << std::setw(8) << std::setfill('0') << index;
        for (int i = 0; i < 8; i++)
        {
            if (index + i < len)
            {
                out << " " << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<uint32_t>(data[i]);
            }
            else
            {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 8; i < 16; i++)
        {
            if (index + i < len)
            {
                out << " " << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<uint32_t>(data[i]);
            }
            else
            {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 0; i < 16; i++)
        {
            char ch = (index + i < len ? data[i] : ' ');
            out << ((ch < 32) ? '.' : ch);
        }
        out << "\n";
        data += 16;
        index += 16;
    }
    out.flags(flags);
}

std::string ngraph::to_lower(const std::string& s)
{
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::tolower);
    return rc;
}

std::string ngraph::to_upper(const std::string& s)
{
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::toupper);
    return rc;
}

string ngraph::trim(const string& s)
{
    string rc = s;
    // trim trailing spaces
    size_t pos = rc.find_last_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(0, pos + 1);
    }

    // trim leading spaces
    pos = rc.find_first_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(pos);
    }
    return rc;
}

vector<string> ngraph::split(const string& src, char delimiter, bool do_trim)
{
    size_t pos;
    string token;
    size_t start = 0;
    vector<string> rc;
    while ((pos = src.find(delimiter, start)) != std::string::npos)
    {
        token = src.substr(start, pos - start);
        start = pos + 1;
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    if (start <= src.size())
    {
        token = src.substr(start);
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    return rc;
}

size_t ngraph::hash_combine(const std::vector<size_t>& list)
{
    size_t seed = 0;
    for (size_t v : list)
    {
        seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

void* ngraph::aligned_alloc(size_t alignment, size_t size)
{
#ifdef __APPLE__
    return new uint64_t[round_up(size, sizeof(uint64_t)) / sizeof(uint64_t)];
#else
    return ::aligned_alloc(alignment, size);
#endif
}

void ngraph::aligned_free(void* p)
{
#ifdef __APPLE__
    delete[] reinterpret_cast<uint64_t*>(p);
#else
    free(p);
#endif
}

size_t ngraph::round_up(size_t size, size_t alignment)
{
    if (alignment == 0)
    {
        return size;
    }

    size_t remainder = size % alignment;
    if (remainder == 0)
    {
        return size;
    }

    return size + alignment - remainder;
}

ngraph::FpropCache ngraph::cache_fprop(std::shared_ptr<ngraph::Function> fprop,
                                       std::shared_ptr<ngraph::Function> bprop)
{
    using namespace ngraph;

    // Create a fprop_cache object to store the results of this analysis
    FpropCache fprop_cache;
    fprop_cache.node_param_map = std::make_shared<NodeMap>();

    // Traverse bprop to find all of the nodes in the bprop graph
    std::unordered_set<std::shared_ptr<Node>> in_bprop;
    ngraph::traverse_nodes(bprop, [&in_bprop](std::shared_ptr<Node> node) {
        if (node->get_outputs().size() == 1)
        {
            if (in_bprop.count(node) == 0)
            {
                in_bprop.insert(node);
            }
        }
    });

    // Traverse fprop to make a map that stores parameters with the same
    // shape and element type as the nodes in fprop iff they are in bprop
    // and aren't inputs to bprop
    auto bprop_inputs = bprop->get_parameters();
    ngraph::traverse_nodes(
        fprop, [&fprop_cache, &in_bprop, &bprop_inputs](std::shared_ptr<Node> node) {
            if (in_bprop.count(node) != 0 &&
                std::find(bprop_inputs.begin(), bprop_inputs.end(), node) == bprop_inputs.end())
            {
                fprop_cache.node_param_map->add(
                    node,
                    std::make_shared<op::Parameter>(node->get_element_type(), node->get_shape()));
            }
        });

    // clone the nodes in bprop, replacing fprop-related nodes with the
    // intermediate parameters from fprop_cache. This breaks connections in the
    // bprop graph such that only intermediate values from fprop needed by bprop
    // are still connected to the bprop graph as parameters
    ngraph::clone_nodes(bprop->get_ops(), *(fprop_cache.node_param_map));

    // invert the fprop_cache cloned node map for easy back and for acces.
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> inverted_node_map;
    for (auto kv : fprop_cache.node_param_map->get_node_map())
    {
        inverted_node_map[kv.second] = kv.first;
    }

    // get cloned bprop results
    ResultVector cloned_results;
    NodeVector result_nodes;
    for (auto node : bprop->get_results())
    {
        auto result = std::dynamic_pointer_cast<op::Result>(fprop_cache.node_param_map->get(node));
        if (!result)
        {
            throw ngraph_error("Expected op::Result values for op::Result keys in node_param_map");
        }
        cloned_results.push_back(result);
        result_nodes.push_back(result);
    }

    // Utility for getting bprop parameters with fprop cache.
    auto get_bprop_params = [&bprop_inputs, &fprop_cache]() {
        // get cloned bprop parameters
        op::ParameterVector bprop_input_params;
        for (auto param : bprop_inputs)
        {
            bprop_input_params.push_back(
                std::dynamic_pointer_cast<op::Parameter>(fprop_cache.node_param_map->get(param)));
        }

        // add the cached fprop nodes as inputs to bprop
        for (auto x : fprop_cache.fprop_output_nodes)
        {
            bprop_input_params.push_back(
                std::dynamic_pointer_cast<op::Parameter>(fprop_cache.node_param_map->get(x)));
        }
        return bprop_input_params;
    };

    // Traverse the graph from the cloned results of bprop. If we find a parameter
    // that's not an original input of bprop, this is an intermediate value of
    // fprop that needs to be returned from fprop and send to bprop
    auto cloned_bprop_inputs = get_bprop_params();
    ngraph::traverse_nodes(
        result_nodes,
        [&cloned_bprop_inputs, &fprop_cache, &inverted_node_map](std::shared_ptr<Node> node) {
            auto pnode = std::dynamic_pointer_cast<op::Parameter>(node);
            if (pnode != nullptr &&
                std::find(cloned_bprop_inputs.begin(), cloned_bprop_inputs.end(), pnode) ==
                    cloned_bprop_inputs.end())
            {
                fprop_cache.fprop_output_nodes.push_back(inverted_node_map.at(node));
            }
        });

    // create the new outputs for fprop and the new fprop function
    ResultVector fprop_outputs = fprop->get_results();

    for (auto fpir : fprop_cache.fprop_output_nodes)
    {
        if (std::dynamic_pointer_cast<op::Result>(fpir))
        {
            throw ngraph_error("Expected op::Result in fprop->get_results()");
        }
        fprop_outputs.push_back(std::make_shared<op::Result>(fpir));
    }

    fprop_cache.fprop = std::make_shared<Function>(fprop_outputs, fprop->get_parameters());

    // Create the new bprop function with cloned results and cached parameters.
    fprop_cache.bprop = std::make_shared<Function>(cloned_results, get_bprop_params());

    return fprop_cache;
}

size_t stopwatch::get_call_count() const
{
    return m_total_count;
}

size_t stopwatch::get_seconds() const
{
    return chrono::duration_cast<chrono::seconds>(get_timer_value()).count();
}

size_t stopwatch::get_milliseconds() const
{
    return chrono::duration_cast<chrono::milliseconds>(get_timer_value()).count();
}

size_t stopwatch::get_microseconds() const
{
    return chrono::duration_cast<chrono::microseconds>(get_timer_value()).count();
}

size_t stopwatch::get_nanoseconds() const
{
    return get_timer_value().count();
}

chrono::nanoseconds stopwatch::get_timer_value() const
{
    if (m_active)
    {
        return (m_clock.now() - m_start_time);
    }
    else
    {
        return m_last_time;
    }
}

size_t stopwatch::get_total_seconds() const
{
    return chrono::duration_cast<chrono::seconds>(m_total_time).count();
}

size_t stopwatch::get_total_milliseconds() const
{
    return chrono::duration_cast<chrono::milliseconds>(m_total_time).count();
}

size_t stopwatch::get_total_microseconds() const
{
    return chrono::duration_cast<chrono::microseconds>(m_total_time).count();
}

size_t stopwatch::get_total_nanoseconds() const
{
    return m_total_time.count();
}

namespace ngraph
{
    template <>
    float parse_string<float>(const std::string& s)
    {
        const char* tmp = s.c_str();
        char* end;
        float result = strtof(tmp, &end);
        if (*end != 0)
        {
            throw std::runtime_error("Could not parse literal '" + s + "'");
        }
        return result;
    }

    template <>
    double parse_string<double>(const std::string& s)
    {
        const char* tmp = s.c_str();
        char* end;
        double result = strtod(tmp, &end);
        if (*end != 0)
        {
            throw std::runtime_error("Could not parse literal '" + s + "'");
        }
        return result;
    }
}

std::ostream& operator<<(std::ostream& os, const ngraph::NodeVector& nv)
{
    std::vector<std::string> names;
    for (auto n : nv)
    {
        names.push_back(n->get_name());
    }
    os << vector_to_string(names);
    return os;
}

void ngraph::check_fp_values_isinf(const char* name, const float* array, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (std::isinf(array[i]))
        {
            throw std::runtime_error("Discovered Inf in '" + string(name) + "'");
        }
    }
}

void ngraph::check_fp_values_isinf(const char* name, const double* array, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (std::isinf(array[i]))
        {
            throw std::runtime_error("Discovered Inf in '" + string(name) + "'");
        }
    }
}

void ngraph::check_fp_values_isnan(const char* name, const float* array, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (std::isinf(array[i]))
        {
            throw std::runtime_error("Discovered NaN in '" + string(name) + "'");
        }
    }
}

void ngraph::check_fp_values_isnan(const char* name, const double* array, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (std::isinf(array[i]))
        {
            throw std::runtime_error("Discovered NaN in '" + string(name) + "'");
        }
    }
}

template <typename T>
T ngraph::apply_permutation(T input, AxisVector order)
{
    if (input.size() != order.size())
    {
        throw "input and order sizes don't match!";
    }

    T output(input.size());

    for (size_t i = 0; i < order.size(); i++)
    {
        output[i] = input.at(order.at(i));
    }

    return output;
}

template AxisVector ngraph::apply_permutation<AxisVector>(AxisVector input, AxisVector order);
template Shape ngraph::apply_permutation<Shape>(Shape input, AxisVector order);

AxisVector ngraph::get_default_order(const Shape& shape)
{
    return get_default_order(shape.size());
}

AxisVector ngraph::get_default_order(size_t rank)
{
    AxisVector default_order(rank);
    std::iota(begin(default_order), end(default_order), 0);
    return default_order;
}
