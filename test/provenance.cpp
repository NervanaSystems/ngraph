//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/provenance.hpp"

using namespace std;
using namespace ngraph;
using ::testing::Return;

using ProvSet = std::unordered_set<std::string>;

TEST(provenance, provenance)
{
    class ProvenanceEnabler
    {
    public:
        ProvenanceEnabler()
        {
            saved_enable_state = get_provenance_enabled();
            set_provenance_enabled(true);
        }
        ~ProvenanceEnabler() { set_provenance_enabled(saved_enable_state); }
    private:
        bool saved_enable_state;
    } provenance_enabler;

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    // Replacement:
    //
    //       A{tag_a} B{tag_b}
    //              | |
    //         C := D{}
    //
    // After:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        D{tag_c}
    //
    // Comment:
    //   * D is the replacement root, and its insertion kills C. We should not, however, consider
    //     A and B to be killed, because they are not post-dominated by D until after C is cut out
    //     of the graph.
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto new_c = make_shared<op::Subtract>(a, b);
        replace_node(c, new_c);

        EXPECT_EQ(new_c->get_provenance_tags(), ProvSet{"tag_c"});
    }

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    // Replacement:
    //
    //
    //
    //     A{tag_a}  B{tag_b}
    //        |      |
    //   C -> D{tag_d}
    //
    // After:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        D{tag_c,tag_d}
    //
    // Comment:
    //   * D is the replacement root, and its insertion kills C. We should not, however, consider
    //     A and B to be killed, because they are not post-dominated by D until after C is cut out
    //     of the graph.
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto d = make_shared<op::Subtract>(a, b);
        d->add_provenance_tag("tag_d");
        replace_node(c, d);

        EXPECT_EQ(d->get_provenance_tags(), (ProvSet{"tag_c", "tag_d"}));
    }

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    // Replacement:
    //
    //   C -> D{tag_d}
    //
    // After:
    //
    //   D{tag_a,tag_b,tag_c,tag_d}
    //
    // Comment:
    //   * D is the replacement root, and its insertion kills A, B, and C.
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto d = make_zero(element::i32, Shape{2, 3, 4});
        d->add_provenance_tag("tag_d");
        replace_node(c, d);

        EXPECT_EQ(d->get_provenance_tags(), (ProvSet{"tag_a", "tag_b", "tag_c", "tag_d"}));
    }

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    // Replacement:
    //
    //   C -> D{}
    //
    // After:
    //
    //   D{tag_a,tag_b,tag_c}
    //
    // Comment:
    //   * D is the replacement root, and its insertion kills A, B, and C.
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto d = make_zero(element::i32, Shape{2, 3, 4});
        replace_node(c, d);

        EXPECT_EQ(d->get_provenance_tags(), (ProvSet{"tag_a", "tag_b", "tag_c"}));
    }

    //
    // Before:
    //
    //   A{tag_a}  B{tag_b}
    //         |   |
    //        C{tag_c}
    //
    //
    // Replacement:
    //
    //   A{tag_a}  B{tag_b}
    //         |      |
    //       E{tag_e} |
    //           |    |
    //     C -> D{tag_d}
    //
    //
    // After:
    //
    //   A{tag_a}  B{tag_b}
    //         |      |
    //       E{tag_e} |
    //           |    |
    //          D{tag_c, tag_d}
    //
    // Comment:
    //   * D is the replacement root replacing C and creating a new argument node E
    //
    {
        auto x = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});
        auto y = make_shared<op::Parameter>(element::i32, PartialShape{2, 3, 4});

        auto a = make_shared<op::Add>(x, y);
        a->add_provenance_tag("tag_a");
        auto b = make_shared<op::Multiply>(y, x);
        b->add_provenance_tag("tag_b");
        auto c = make_shared<op::Subtract>(a, b);
        c->add_provenance_tag("tag_c");

        auto f = make_shared<Function>(c, ParameterVector{x, y});

        auto e = make_shared<op::Subtract>(a, x);
        auto d = make_shared<op::Subtract>(e, b);
        d->add_provenance_tag("tag_d");

        replace_node(c, d);

        EXPECT_EQ(d->get_provenance_tags(), (ProvSet{"tag_c", "tag_d"}));
    }
}
