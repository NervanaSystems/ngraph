.. fusion/passes-that-use-matcher.rst:


Passes that use Matcher
=======================

* CPUFusion (GraphRewrite)
* CoreFusion (GraphRewrite)
* ReshapeElimination (GraphRewrite)
* AlgebraicSimplification
* CPUPostLayoutOptimizations (GraphRewrite)
* CPURnnMatFusion
* and many more...



Register ``simplify_neg`` handler
----------------------------------


.. code-block:: cpp

   static std::unordered_map<std::type_index, std::function<bool(std::shared_ptr<Node>)>>
            initialize_const_values_to_ops()
        {
            return std::unordered_map<std::type_index, std::function<bool(std::shared_ptr<Node>)>>({
                {TI(op::Add), simplify_add},
                {TI(op::Multiply), simplify_multiply},
                {TI(op::Sum), simplify_sum},
                {TI(op::Negative), simplify_neg}
            });
        }

Add a fusion 
~~~~~~~~~~~~

``max(0, A) = Relu(A)`` 



Pattern for capturing 
~~~~~~~~~~~~~~~~~~~~~

|image11|

``max(0, A) = Relu(A)``  

.. code-block:: cpp

   namespace ngraph
    {
        namespace pass
        {
            class CoreFusion;
        }
    }
    
    class ngraph::pass::CoreFusion : public ngraph::pass::GraphRewrite
    {
    public:
        CoreFusion()
            : GraphRewrite()
        {
            construct_relu_pattern();
        }

        //this should go in a cpp file.
        void construct_relu_pattern()
        {
            auto iconst0 = ngraph::make_zero(element::i32, Shape{});
            auto val = make_shared(iconst0);
            auto zero = make_shared(iconst0, nullptr, NodeVector{iconst0});

            auto broadcast_pred = [](std::shared_ptr n) {
                return static_cast(std::dynamic_pointer_cast(n));
            };
            auto skip_broadcast = std::make_shared(zero, broadcast_pred);
            auto max = make_shared(skip_broadcast, val);

        pattern::graph_rewrite_callback callback = [val, zero](pattern::Matcher& m) { 
                NGRAPH_DEBUG << "In a callback for construct_relu_pattern against "
                            << m.get_match_root()->get_name();

                auto pattern_map = m.get_pattern_map();
                auto mzero = m.get_pattern_map()[zero];
                if (!ngraph::is_zero(mzero))
                {
                    NGRAPH_DEBUG << "zero constant = " << mzero->get_name() << " not equal to 0n";
                    return false;
                }
                auto mpattern = m.get_match_root();

                auto cg = shared_ptr(new op::Relu(pattern_map[val]));
                ngraph::replace_node(m.get_match_root(), cg);
                return true;
            };

            auto m = make_shared(max, callback); 
            this->add_matcher(m);
        }
    };
            
Recurrent patterns 
------------------

Equivalent to ``"A(BC)+A"`` in regexes 


``(((A + 0) + 0) + 0) = A``

|image12|

|image13|


.. code-block:: cpp

   Shape shape{};
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto rpattern = std::make_shared<pattern::op::Label>(b);
    auto iconst0 = ngraph::make_zero(element::i32, shape);
    auto abs = make_shared<op::Abs>(a);
    auto add1 = iconst0 + b;
    auto add2 = iconst0 + add1;
    auto add3 = iconst0 + add2;
    auto padd = iconst0 + rpattern;
    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    RecurrentMatcher rm(padd, rpattern, empty_correlated_matches, nullptr);
    ASSERT_TRUE(rm.match(add3));



.. |image11| image:: mg/fusion_pattern.png
.. |image12| image:: mg/rp_graph1.png
.. |image13| image:: mg/rp_pattern.png