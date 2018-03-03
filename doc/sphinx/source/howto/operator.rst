.. operator.rst

############################
Build a graph with operators
############################

This section illustrates the use of C++ operators to simplify the
building of graphs.

Several C++ operators have been overloaded so that when given nodes as
arguments they construct a node that performs the operation on the
nodes. For example, the following:

.. literalinclude:: ../../../examples/abc.cpp
   :language: cpp
   :lines: 32-32

can be simplified to:	   

.. literalinclude:: ../../../examples/abc_operator.cpp
   :language: cpp
   :lines: 31

The expression ``a + b`` is equivalent to
``std::make_shared<op::Add>(a, b)`` and the ``*`` operator similarly
returns ``std::make_shared<op::Multiply>`` to its arguments.
