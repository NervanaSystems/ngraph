.. optimize/translate.rst: 


Translating graphs with nGraph |trade| Compiler
===============================================

* :ref:`about_translate`
* :ref:`pattern_matching`
* :ref:`graph_rewrite`


.. _about_translate: 

About 
-----

The nGraph Compiler is an *optimizing* compiler. As such, it performs a series 
of optimization passes over a given function graph to translate it into a 
different graph that is not only semantically equivalent to the original, but 
also inherently optimized with superior runtime characteristics for any of 
nGraph's current or future backends. Indeed, the ability to increase training 
performance or reduce inference latency by simply adding another device of *any* 
form factor -- compute, ASIC, or VPU -- is one of the key benefits available to 
any framework that integrates the nGraph Library.  

In handling a :term:`function graph`, there are many ways to describe what 
happens when we translate the framework's output of ops into an nGraph 
graph. *Fusing* is the term we will use here, but it also can be described as: 
*combining*, *folding*, *collapsing*, or *merging* graph functions.  One common 
use case is to fuse a subgraph from the function graph into 
:doc:`one of the nGraph Core ops <../ops/index>`. In other words, nGraph Compiler 
will take a sub-set of computations from the graph and make it more efficient via 
:term:`fusion`.

The optimization passes may include algebraic simplifications, domain-specific 
simplifications, and fusion. Most passes share the same mode of operation (or 
the same operational structure) and consist of two stages:

#. Locating a list of potential transformation candidates (usually, subgraphs) 
   in the given graph.
#. Transforming the selected candidates into semantically-equivalent subgraphs 
   that (usually) run faster.

Let's consider an example. A user would like to execute a simple graph that 
describes the following arithmetic expression:

:math:`a + b * 1` or :math:`Add(a, Mul(b, 1))` 

In the above expressions, `1` is an identity element: any element 
multiplied by the identity element is equal to itself. This is the same as saying

:math:`b * 1 = b` 

The writer of an algebraic-simplification pass would probably want to ``locate`` 
all multiplication expressions where multiplicands are multiplied by `1` (for 
stage 1) and ``transform``, `` simplify``, or ``replace`` those expressions with 
just their multiplicands (for stage 2).

To make the work of an optimization pass writer easier, the nGraph library 
includes facilities that enable the *finding* of relevant candidates using pattern 
matching (via `pattern/matcher.hpp`), and the *transforming* of the original graph 
into a condensed version ( via `pass/graph_rewrite.hpp`).

Let's consider each of the two in more detail and many ways they can help the 
work of the optimization pass writer.


.. _pattern_matching: 

Applying pattern matcher to fuse ops
-------------------------------------

Before delving into the details of pattern matching, note that the sole purpose 
of the pattern-matching step is to **find** patterns in the framework's graph of 
outputs.  A subgraph, then, could contain any operation nGraph's IR defines 
(addition, subtraction, etc) along with some special wildcard nodes. 

An analogy that is probably familiar to many programmers is that of the ``Regex``, 
AKA "Regular expression". Just as one can write a regex (pattern) and run it 
through some text to find and/or replace the occurences of that pattern in 
the given text, so too can one write optimization-passes to construct patterns 
which are just regular nGraph graphs and run those patterns through given graphs.

In the ``Regex`` analogy, 

* Letters (for example: `A`, `B`, `m`, `n`), 
* Strings (for example: `ABBA`, `Hello world`), and
* Collective Symbols (for some regex programs they are `*`, `.`)

respectfully correspond to: 

.. Letters
* ``ngraph::Node`` (``op::Add``, ``op::BatchNormBackprop``), 
.. Strings
* Graphs consisting of ``ngraph::Nodes``
.. Collective Symbols 
* ``op::*`` (for some graph programs they are ``pattern::op::Label``, ``pattern::op::Skip``)

where Operators need arguments, and Leaves cannot take arguments.  


At the lower level, the nGraph C++ API looks like this:  

.. doxygenclass:: ngraph::pattern::Matcher
   :project: ngraph
   :members:


To create a trivial graph representing  ``-(-A) = A``

.. code-block:: cpp 

   auto a = make_shared(element::i32, shape); 
   auto neg1 = make_shared(a); 
   auto neg2 = make_shared(neg1);

|image1|



For exact pattern matching

.. code-block:: cpp 

   auto a = make_shared<op::Parameter>(element::i32, shape);
   auto neg1 = make_shared<op::Negative>(a);
   auto neg2 = make_shared<op::Negative>(neg1);



|image2|


.. _graph_rewrite:

Using ``GraphRewrite`` to fuse ops
-----------------------------------



.. MOARRR complex graph w/ $-(-A) = A$ \`\`\`cpp auto
	a = make\_shared(element::i32, shape); auto absn = make\_shared(a); auto
	neg1 = make\_shared(absn); auto neg2 = make\_shared(neg1); \`\`\` ---
	### MOARRR complex graph w/ $-(-A) = A$ |image3| --- ### Even MOARRR
	complex graph w/ $-(-A) = A$ \`\`\`cpp auto a =
	make\_shared(element::i32, shape); auto b = make\_shared(element::i32,
	shape); auto c = a + b; auto absn = make\_shared(c); auto neg1 =
	make\_shared(absn); auto neg2 = make\_shared(neg1); \`\`\` --- ### Even
	MOARRR complex graph w/ $-(-A) = A$ |image4| --- ### \`Label\` a.k.a.
	"\`.\`" in regexes \`\`\`cpp //note element::f32, will still match
	integer Graph1 and Graph2 auto lbl = std::make\_shared(element::f32,
	Shape{}); auto neg1 = make\_shared(lbl); auto neg2 = make\_shared(neg1);
	\`\`\` --- ### Pattern matching $-(-A) = A$ |image5| --- ###
	Constructing labels from existing nodes \`\`\`cpp auto a =
	make\_shared(element::i32, shape); //\`lbl\` borrows the type and shape
	information from \`a\` auto lbl = std::make\_shared(a); auto neg1 =
	make\_shared(a); auto neg2 = make\_shared(neg1); \`\`\` --- ### Problem
	1.a ### $-(-A) = A, A ::= \\\\{op::Add,op::Sub\\\\}$ --- ### Double
	Negative w/ Add |image6| --- ### Double Negative w/ Sub |image7| ---
	\`\`\`cpp //predicates are of type std::function)> auto add\_or\_sub =
	[](std::shared\_ptr n) { return std::dynamic\_pointer\_cast(n) !=
	nullptr \|\| std::dynamic\_pointer\_cast(n) != nullptr }; // auto lbl =
	std::make\_shared( element::f32, Shape{}, add\_or\_sub ); auto neg1 =
	make\_shared(a); auto neg2 = make\_shared(neg1); \`\`\` --- ### Problem
	2 ### $A + 0 = A$ ### $A + Broadcast(0) = A$ --- ### Equivalent to
	"A0B?C" in regexes --- ### $A + 0 = A$ |image8| --- ### $A +
	Broadcast(0) = A$ |image9| --- ### \`Skip\` a.k.a. "?" in regexes
	\`\`\`cpp auto iconst = ngraph::make\_zero(element::i32, Shape{}); auto
	label = std::make\_shared(iconst); //Predicate tells Matcher the node
	type(s) it should expect auto bcst\_pred = [](std::shared\_ptr n) {
	return std::dynamic\_pointer\_cast(n) != nullptr; }; auto bcst\_skip =
	std::make\_shared(iconst, bcst\_pred); //Matcher is aware that
	\`op::Add\` is commutative //Also matches //$0 + A = A$ //$Broadcast(0)
	+ A = A$ auto add = std::make\_shared(label, bcst\_skip); \`\`\` --- ###
	Pattern matching $Broadcast(0) + A = A$ and $0 + A = A$ |image10| ---
	### \`ngraph::pattern::Matcher\` \`\`\`cpp //create a matcher object
	auto matcher = std::make\_shared( neg\_pattern //std::shared\_ptr ,
	nullptr //callback ); if (matcher.match(graph1)) { std::cout << "root =
	" << matcher.get\_match\_root()->get\_name() << std::endl; std::cout <<
	"lbl = " << matcher.get\_pattern\_map[lbl]->get\_name() << std::endl; }
	\`\`\` --- ### Passes that use \`Matcher\` \* \`CPUFusion\`
	(\`GraphRewrite\`) \* \`CoreFusion\` (\`GraphRewrite\`) \*
	\`ReshapeElimination\` (\`GraphRewrite\`) \* \`AlgebraicSimplification\`
	\* \`CPUPostLayoutOptimizations\` (\`GraphRewrite\`) \*
	\`CPURnnMatFusion\` --- ### Add a transformation for $-(-A) = A$

::

..    static bool simplify_neg(std::shared_ptr n)
            {
                NGRAPH_DEBUG << "In simplify_add for simplify_neg" << n->get_name();
                auto lbl = std::make_shared(element::f32, Shape{});
                auto neg1 = std::make_shared(lbl);
                auto neg2 = std::make_shared(neg1);
                //Create a new matcher capturing `neg2` pattern
                auto matcher = std::make_shared(neg2);
            
                if (matcher->match(neg2))
                {
                    //Extract the node bound to `lbl`
                    auto m_lbl = matcher->get_pattern_map[lbl];
                    NGRAPH_DEBUG << "Replacing " 
                        << n->get_name() << " with " << m_lbl->get_name();
                    //Replace `n` with `m_lbl`
                    ngraph::replace_node(n, m_lbl);
                }
            }
        

.. --- ### Register \`simplify\_neg\` handler
::

..    static std::unordered_map)>>
            initialize_const_values_to_ops()
        {
            return std::unordered_map)>>({
                {TI(op::Add), simplify_add},
                {TI(op::Multiply), simplify_multiply},
                {TI(op::Sum), simplify_sum},
                {TI(op::Negative), simplify_neg}
            });
        }

.. --- ### Add a fusion ### $max(0, A) = Relu(A)$ --- ### Pattern for capturing $max(0, A) = Relu(A)$ |image11| ---
::

..            namespace ngraph
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
                void construct_relu_pattern();
            };
            

---
::

    void pass::CoreFusion::construct_relu_pattern()
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
                NGRAPH_DEBUG << "zero constant = " << mzero->get_name() << " not equal to 0\n";
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

--- ### Recurrent patterns $ (((A + 0) + 0) + 0) = A$ --- ### Equivalent
to "A(BC)+A" in regexes --- ### $ (((A + 0) + 0) + 0) = A$ |image12| ---
### \`Label\` + 0 |image13| ---
::

    Shape shape{};
    auto a = make_shared(element::i32, shape);
    auto b = make_shared(element::i32, shape);
    auto rpattern = std::make_shared(b);
    auto iconst0 = ngraph::make_zero(element::i32, shape);
    auto abs = make_shared(a);
    auto add1 = iconst0 + b;
    auto add2 = iconst0 + add1;
    auto add3 = iconst0 + add2;
    auto padd = iconst0 + rpattern;
    std::set> empty_correlated_matches;
    RecurrentMatcher rm(padd, rpattern, empty_correlated_matches, nullptr);
    ASSERT_TRUE(rm.match(add3));


.. |image1| image:: mg/pr1_graph1.png
.. |image2| image:: mg/pr1_pattern.png
.. |image3| image:: mg/pr1_graph2.png
.. |image4| image:: mg/pr1_graph3.png
.. |image5| image:: mg/pr1_pattern2.png
.. |image6| image:: mg/pr1_graph4.png
.. |image7| image:: mg/pr1_graph5.png
.. |image8| image:: mg/pr2_graph1.png
.. |image9| image:: mg/pr2_graph2.png
.. |image10| image:: mg/pr2_pattern2.png
.. |image11| image:: mg/fusion_pattern.png
.. |image12| image:: mg/rp_graph1.png
.. |image13| image:: mg/rp_pattern.png


