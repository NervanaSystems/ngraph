[PR #276](https://github.com/NervanaSystems/private-ngraph-cpp/pull/276) ; **9d9e43b9838c0c8c57dad25c8294676f270aa008**

* API changes:

    * `Tuple`,`GetTupleElement`,`Function` were renamed to `XLATuple`, `XLAGetTupleElement`, `XLAFunction`. The bridges need to be updated to use the new types.
    * New types `Function`,`GetOutputElement` and APIs (e.g. `Function::get_results`, `Function::get_return_types`, `Function::Function`, etc) were added to fulfill a design objective of removing tuples in favour of flattened collections. The bridge developers will be responsible for converting `XLATuple`, `XLAGetTupleElement` into `Function` and `GetOutputElement`

* New Features:

* Ngraph++ Core Notes:
    * `get_arguments` is replaced with `get_arguments_via_inputs` which will be deprecated in the nearest future. `get_inputs` and `get_outputs` should be used instead. These changes are the first steps toward removing `m_args` and using I/O descriptors for graph traversals. 

* Resolved issues
* Known issues
* Documentation changes

