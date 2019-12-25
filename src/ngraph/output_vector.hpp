#include <memory>
#include <vector>

namespace ngraph
{
    class Node;

    template <typename T>
    class Output;

    using NodeVector = std::vector<std::shared_ptr<Node>>;
    using OutputVector = std::vector<Output<Node>>;
}