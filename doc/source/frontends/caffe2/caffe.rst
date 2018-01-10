.. caffe.rst:

.. ---------------------------------------------------------------------------
.. Copyright 2017 Intel Corporation
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Caffe\*
=======

An Intel® nGraph™ abstraction layer includes utilities that enable frontend 
interoperability with frameworks such as `caffe2`_.  

The Caffe\* importer lets users build a graph of Intel nGraph ``ops`` from 
the layers in model prototxt. This graph can then be executed using 
transformers from the Intel nGraph API.


Sum example
-----------

Here's a sample sum example for the caffe importer.

The sample prototxt is given below to compute the operation **D = A+B+C**::

    name: "Sum"
    layer {
        name: "A"
        type: "DummyData"
        top: "A"
        dummy_data_param {
            data_filler {
                type: "constant"
                value: 1.0
            }
            shape {
                dim:2
                dim:3
            }
        }
    }

    layer {
        name: "B"
        type: "DummyData"
        top: "B"
        dummy_data_param {
            data_filler {
                type: "constant"
                value: 3.0
            }
            shape {
                dim:2
                dim: 3
            }
        }
    }

    layer {
        name: "C"
        type: "DummyData"
        top: "C"
        dummy_data_param {
            data_filler {
                type: "constant"
                value: -2.0
            }
            shape {
                dim:2
                dim:3 
            }
        }
    }

    layer {
        name: "D"
        type: "Eltwise"
        top: "D"
        bottom: "A"
        bottom: "B"
        bottom: "C"
        eltwise_param {
            operation: SUM
        }
    }



The following sample script computes ``D`` for the above prototxt:


.. literalinclude:: ../../../../ngraph/frontends/caffe/examples/sum.py
   :language: python
   :linenos:
   :lines: 16-28
   :caption: /ngraph/frontends/caffe/examples/sum.py

In above script: 

* Line 3 imports the parsing functionality available through the nGraph library,  
* ``parse_prototxt()`` reads the prototxt and outputs a graph, 
* ``op = op_map.get("D")`` demonstrates that the ngraph ``op`` of any layer 
  can be obtained from the .. py:function::`get()` function on the graph
* ``res = ngt.make_transformer().computation(op)()`` is the code that makes use
  of the transformers from the nGraph backend and executes the ngraph of the layer.


Command-line interface
----------------------

A caffe-like :abbr:`Command Line Interface (CLI)` is also available to run the 
prototxt, as shown below::

.. code-block:: console

    $ python importer.py compute -model  sum.prototxt -name C,D,A 


Limitations
------------

Currently only sum operations on dummy data can be executed. Stay tuned for more 
functionality in future releases. 



.. _caffe2: http://caffe.berkeleyvision.org
.. _prototxt: https://stackoverflow.com/questions/37418370/how-to-write-comments-in-prototxt-files

