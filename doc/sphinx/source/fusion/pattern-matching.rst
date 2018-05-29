.. fusion/pattern-matching.rst: 

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

.. Letter -> Node

* ``ngraph::Node`` (``op::Add``, ``op::BatchNormBackprop``), 

.. Strings

* Graphs consisting of ``ngraph::Nodes``, and

.. Collective Symbols 

* ``op::*`` (for some graph programs they are ``pattern::op::Label``, ``pattern::op::Skip``)

where Operators need arguments, and Leaves cannot take arguments.  


At the lower level, the nGraph C++ API is:  

.. doxygenclass:: ngraph::pattern::Matcher
   :project: ngraph
   :members:


To create a trivial graph representing ``-(-A) = A``:

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





.. |image1| image:: mg/pr1_graph1.png
.. |image2| image:: mg/pr1_pattern.png
