// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

class StubBackend : public runtime::Backend
{
public:
    std::shared_ptr<ngraph::runtime::TensorView>
        create_tensor(const ngraph::element::Type& element_type, const Shape& shape)
    {
        return nullptr;
    }

    /// @brief Return a handle for a tensor for given mem on backend device
    std::shared_ptr<ngraph::runtime::TensorView> create_tensor(
        const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer)
    {
        return nullptr;
    }
    bool compile(std::shared_ptr<Function> func) { return false; }
    bool call(std::shared_ptr<Function> func,
              const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
              const std::vector<std::shared_ptr<runtime::TensorView>>& inputs)
    {
        return false;
    }
    bool is_op_supported(const std::string& name, element::Type t) const
    {
        NGRAPH_INFO << name << ", " << t;
        return true;
    }
};

TEST(op, is_op)
{
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    EXPECT_TRUE(arg0->is_parameter());
}

TEST(op, is_parameter)
{
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    auto t0 = make_shared<op::Add>(arg0, arg0);
    ASSERT_NE(nullptr, t0);
    EXPECT_FALSE(t0->is_parameter());
}

TEST(op, is_supported)
{
    runtime::Backend::register_backend("STUB", make_shared<StubBackend>());
    auto backend = runtime::Backend::create("STUB");
    EXPECT_TRUE(backend->is_supported<op::Add>(element::f32));
}
