.. api.rst:

API 
###

.. Don't add Python APIs that will break the build.  

Sections
========

Templatable Functions
----------------------

.. Function template to enable type-checking? Can be like macros, but more
   compact 

.. So-called "overloaded" functions perform similar operations on different 
   types of data. When operations are _identical_ for each type, using a  
   function template might make sense.  

.. Function template definition could take many conceptually-possible forms, 
   more than listed here even
   
.. code-block:: cpp

   template< class ShapeDim >
   // or
   template< typename AxesType >  
   // or
   template< class ShapeDefX, class ShapeDefY, class VectorType  >


Class Templates and Nontype Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example documented here is the ``BinaryElementWise`` op constructor. In 
classic ngraph, use of an arithmetic operator was handled in the constructor 
for ``BinaryElementWiseOp``, which was able to (1) convert scalar inputs to 
different kinds of ops, and (2) to then broadcast shapes appropriately.

.. list-table:: BinaryElementWise
   :widths: 15, 30, 21 
   :header-rows: 1
   :align: center

   * - Core
     - nG ++ Templated Class
     - pynGraph
   * - ``BinaryElementWise``
     -  scalar input
     - ``shapeDim``


Templates and Inheritance
~~~~~~~~~~~~~~~~~~~~~~~~~~

Inheritance can be extracted out at any point in the graph where class-template
specialization has been defined.  


.. Templates and Friends
   ~~~~~~~~~~~~~~~~~~~~~

.. template< class ShapeDim > class X

.. friend void f1(); 
   
   // would make function f1 a friend of every class-template
   // specialization instantiated on line 65
   // friendly functions and classes might be out of scope 

   


   


