#include "graph_rewrite.hpp"
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include "ngraph/log.hpp"
#include "ngraph/pattern/matcher.hpp"
#include <numeric>
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
    const std::vector<std::shared_ptr<pattern::Matcher>>& matchers,
    std::shared_ptr<ngraph::Function> f)
{
    bool rewritten = false;
    for (auto node : nodes)
    {
        for (auto matcher : matchers)
        {
            NGRAPH_DEBUG << "Running matcher " << matcher << " on " << node << " , "
                         << node->get_name() << " , is_output = " << node->is_output();
            if (matcher->match(node))
            {
                NGRAPH_DEBUG << "Matcher " << matcher << " matched " << node << " , "
                             << node->get_name();
                rewritten = true;
                auto result = matcher->process_match();
                if (result)
                {
                    f->replace_node(node, result);
                }
                break; //move onto the next node
            }
        }
    }
    return rewritten;
}

bool ngraph::pass::GraphRewrite::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    return run_matchers_on_nodes_list(f->get_ordered_ops(), m_matchers, f);
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

template <typename T>
static std::vector<T> apply_permutation(std::vector<T> input, ngraph::AxisVector order)
{
	if (input.size() != order.size())
	{
		throw "input and order sizes don't match!";
	}

	std::vector<T> output(input.size());

	for (size_t i = 0; i < order.size(); i++)
	{
		output[i] = input.at(order.at(i));
	}

    return output;
}

void ngraph::pass::CPUFusion::construct_reshapex2_pattern()
{
	auto shape_op = Shape{ 3 };
	auto shape_r1 = Shape{ 1, 3 };
	
	auto op = std::make_shared<pattern::op::Label>(element::f32, shape_op);
	auto reshape1 = std::make_shared<op::Reshape>(op, AxisVector{0}, shape_r1);
	auto reshape2 = std::make_shared<op::Reshape>(reshape1, AxisVector{0,1}, shape_op);
	
	auto callback = [op](pattern::Matcher& m) {
		NGRAPH_DEBUG << "In callback for construct_reshapex2_pattern against node = " << m.match_root()->get_name();
		auto pattern_map = m.get_pattern_map();

		std::shared_ptr<ngraph::Node> nn = nullptr;
		auto gop = pattern_map[op];

		if (gop->get_shape() != m.match_root()->get_shape())
		{
            NGRAPH_DEBUG << "Operand shape doesn't match the shape of the second reshape!";
            NGRAPH_DEBUG << "gop " << gop->get_name() << "shape = " << vector_to_string(gop->get_shape());
            NGRAPH_DEBUG << "match_root " << m.match_root()->get_name() << "shape = " << vector_to_string(m.match_root()->get_shape());
			return nn;
		}

		auto r2 = std::dynamic_pointer_cast<op::Reshape>(m.match_root());
		auto r1 = std::dynamic_pointer_cast<op::Reshape>(r2->get_input_op(0));

		Shape do_r2(r1->get_shape().size());
		std::iota(begin(do_r2), end(do_r2), 0);
		Shape do_r1(gop->get_shape().size());
		std::iota(begin(do_r1), end(do_r1), 0);

        NGRAPH_DEBUG << "r1's i/o = " << vector_to_string(r1->get_input_order()) << "do_r1 = " << vector_to_string(do_r1) ;
        NGRAPH_DEBUG << "r2's i/o = " << vector_to_string(r2->get_input_order()) << "do_r2 = " << vector_to_string(do_r2) ;
		
		if (r1->get_input_order() == do_r1 && r2->get_input_order() == do_r2)
		{
            NGRAPH_DEBUG << "Two reshapes were removed!";
			return gop;
		}

		auto perm1 = apply_permutation(do_r1, r1->get_input_order());
		auto perm2 = apply_permutation(perm1, r2->get_input_order());
		if (perm2 == do_r1)
		{
            NGRAPH_DEBUG << "Two transposes were removed!";
			return gop;
		}

		return nn;
	};
	auto m = std::make_shared<ngraph::pattern::Matcher>(reshape2, callback);
	this->add_matcher(m);
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

    ngraph::pattern::gr_callback_fn callback = [W, x, b](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_gemm_pattern against node = "
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        std::shared_ptr<Node> nn = nullptr;

        auto mpattern = m.match_root();
        if (mpattern->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << mpattern->get_name() << " type is not float!";
            return nn;
        }

        auto dot = mpattern->get_input_op(0);
        if (dot->get_shape().size() != 2)
        {
            NGRAPH_DEBUG << "dot = " << dot->get_name() << " shape is not equal to 2!";
            return nn;
        }

        bool transpose_w = false;
        Shape shape_arg0{pattern_map[W]->get_shape()};
        if (!init_cblas_arg(dot->get_input_op(0), pattern_map[W], transpose_w, shape_arg0))
        {
            return nn;
        }

        bool transpose_x = false;
        Shape shape_arg1{pattern_map[x]->get_shape()};
        if (!init_cblas_arg(dot->get_input_op(1), pattern_map[x], transpose_x, shape_arg1))
        {
            return nn;
        }

        auto cg = std::shared_ptr<Node>(new op::CblasGemm(pattern_map[W],
                                                          pattern_map[x],
                                                          mpattern->get_input_op(1),
                                                          shape_arg0,
                                                          shape_arg1,
                                                          transpose_w,
                                                          transpose_x));
        return cg;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(padd, callback);
    this->add_matcher(m);
}
