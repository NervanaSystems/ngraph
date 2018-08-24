.. tanh.rst:

#####
Tanh
#####

.. code-block:: cpp

   Tanh  // Elementwise hyperbolic tangent operation.

..   /// \brief Elementwise hyperbolic tangent operation.
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                     |
        /// | ----- | --------------------------------- | ----------------------------------------------- |
        /// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                           |
        /// | ---------------------- | ------------------------------------------------------------------------------------- |
        /// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \tanh(\texttt{arg}[i_1,\dots,i_n])\f$ |

..       /// \brief Constructs a hyperbolic tangent operation.
            ///
            /// \param arg Node that produces the input tensor.


