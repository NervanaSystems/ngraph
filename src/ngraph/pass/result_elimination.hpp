#include <iostream>

namespace ngraph
{
    namespace pass
    {
        class ResultCopyElimination;
    }
}

class ngraph::pass::ResultCopyElimination : public ngraph::pass::FunctionPass
{
public:
    ResultCopyElimination()
        : FunctionPass()
    {
    }

    virtual bool run_on_function(std::shared_ptr<ngraph::Function> f) override
    {
        NodeVector optimized_results;
        std::set<std::shared_ptr<Node>> seen;
        for (auto res : f->get_results())
        {
            auto arg = res->get_input_op(0);
            //we need a copy
            if (arg->is_parameter())
            {
                optimized_results.push_back(res);
                continue;
            }

            //TODO: check if broadcast replace op::Result w/ a copy of broadcast node

            //TODO: consider other cases where it's easier to recompute than make a copy

            if (seen.count(arg) == 0)
            {
                optimized_results.push_back(arg);
                seen.insert(arg);
            }
            else
            {
                optimized_results.push_back(res);
            }
        }

        f->set_optimized_results(optimized_results);
        return 1;
    }
};
