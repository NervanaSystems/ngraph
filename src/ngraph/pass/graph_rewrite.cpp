#include "graph_rewrite.hpp"
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include "ngraph/log.hpp"
#include "ngraph/pattern/matcher.hpp"

//cpufusion headers
#include "ngraph/graph_util.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/cblas_gemm.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"

bool ngraph::pass::GraphRewrite::run_matchers_on_nodes_list(
    const std::list<std::shared_ptr<ngraph::Node>>& nodes,
    const std::vector<std::shared_ptr<pattern::Matcher>>& matchers)
{
    bool rewritten = false;
    for (auto node : nodes)
    {
        for (auto matcher : matchers)
        {
            NGRAPH_DEBUG << "Running matcher " << matcher << " on " << node << " , "
                         << node->get_name() << " , is_output = " << node->is_output();
            if (!node->is_output() /*this restriction can be lifted when we find an use case for it*/
                &&
                matcher->match(node))
            {
                NGRAPH_DEBUG << "Matcher " << matcher << " matched " << node << " , "
                             << node->get_name();
                rewritten = true;
                matcher->process_match();
                break; //move onto the next node
            }
        }
    }
    return rewritten;
}

bool ngraph::pass::GraphRewrite::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
    return run_matchers_on_nodes_list(nodes, m_matchers);
}

static bool init_cblas_arg(std::shared_ptr<ngraph::Node> reshape,
                           std::shared_ptr<ngraph::Node> arg,
                           bool& transpose_w,
                           ngraph::Shape& shape_w)
{
    auto r_w = std::dynamic_pointer_cast<ngraph::op::Reshape>(reshape);

    if (!r_w)
    {
        return true; //nth to do; reshape isn't a reshape
    }

    if (r_w->get_shape().size() != 2)
    {
        NGRAPH_DEBUG << "Reshape for " << reshape->get_name() << " doesn't reshape into matrix"
                     << ngraph::vector_to_string(r_w->get_shape());
        return false;
    }

    auto io = r_w->get_input_order();
    if (r_w->get_shape().size() != arg->get_shape().size()) //reshape
    {
        ngraph::AxisVector dio(io.size());
        std::iota(begin(dio), end(dio), 0);

        if (io != dio) //we can't reshape and transpose at the same time
        {
            NGRAPH_DEBUG << "Reshape for " << reshape->get_name() << " is not in default order "
                         << ngraph::vector_to_string(io);
            NGRAPH_DEBUG << "r_w shape = " << ngraph::vector_to_string(r_w->get_shape());
            NGRAPH_DEBUG << "arg shape = " << ngraph::vector_to_string(arg->get_shape());
            return false;
        }

        shape_w = r_w->get_shape();
    }
    else
    {
        if (io == ngraph::AxisVector{1, 0})
        {
            transpose_w = true;
        }
        //otherwise no-op reshape
    }

    return true;
}

void ngraph::pass::CPUFusion::construct_gemm_pattern()
{
    auto shape_w = Shape{2, 4};
    auto shape_x = Shape{4, 1};
    auto shape_b = Shape{1};
    auto shape_dot = Shape{2, 1};

    auto W = std::make_shared<pattern::op::Label>(element::f32, shape_w);
    auto x = std::make_shared<pattern::op::Label>(element::f32, shape_x);

    auto reshape_pred = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::Reshape>(n));
    };

    auto skip_w = std::make_shared<pattern::op::Any>(W, reshape_pred);
    auto skip_x = std::make_shared<pattern::op::Any>(x, reshape_pred);

    auto pdot = std::make_shared<op::Dot>(skip_w, skip_x);
    auto b = std::make_shared<pattern::op::Label>(element::f32, shape_b);
    auto pbroadcast = std::make_shared<op::Broadcast>(b, shape_dot, AxisSet{0});
    auto padd = pdot + pbroadcast;

    auto callback = [W, x, b](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_gemm_pattern against node = "
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto mpattern = m.match_root();
        if (mpattern->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << mpattern->get_name() << " type is not float!";
            return;
        }

        auto dot = mpattern->get_input_op(0);
        if (dot->get_shape().size() != 2)
        {
            NGRAPH_DEBUG << "dot = " << dot->get_name() << " shape is not equal to 2!";
            return;
        }

        bool transpose_w = false;
        Shape shape_arg0{pattern_map[W]->get_shape()};
        if (!init_cblas_arg(dot->get_input_op(0), pattern_map[W], transpose_w, shape_arg0))
        {
            return;
        }

        bool transpose_x = false;
        Shape shape_arg1{pattern_map[x]->get_shape()};
        if (!init_cblas_arg(dot->get_input_op(1), pattern_map[x], transpose_x, shape_arg1))
        {
            return;
        }

        auto cg = std::make_shared<op::CblasGemm>(pattern_map[W],
                                                  pattern_map[x],
                                                  mpattern->get_input_op(1),
                                                  shape_arg0,
                                                  shape_arg1,
                                                  transpose_w,
                                                  transpose_x);
        ngraph::replace_node(m.match_root(), cg);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(padd, callback);
    this->add_matcher(m);
}
