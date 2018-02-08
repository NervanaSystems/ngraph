
Elementwise absolute value operation.

 ## Inputs

 |       | Type                              | Description                                     |
 | ----- | --------------------------------- | ----------------------------------------------- |
 | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |

 ## Output

 | Type                   | Description                                                                      |
 | ---------------------- | -------------------------------------------------------------------------------- |
 | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = |\texttt{arg}[i_1,\dots,i_n]|\f$ |
        


Constructs an absolute value operation.
    
Node that produces the input tensor.


            Abs(const std::shared_ptr<Node>& arg)
                : UnaryElementwiseArithmetic("Abs", arg)
            {
            }

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 1)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<Abs>(new_args.at(0));
            }

     
