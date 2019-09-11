# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""Writer classes"""


from opgen.spec.ast import (OpArgument, OpAttribute, OpResult, OpClass)
from opgen.spec.parser import SpecParser

import json
import sys
import argparse
import os

# For conversion from CamelCase to snake_case.
import inflection


class OpArgumentWriter():
    def __init__(self, arg, idx):
        self._name = arg.name
        self._description = arg.description
        self._idx = idx

    def ctor_parameter_proto(self):
        s = ''
        s += 'const ::ngraph::Output<::ngraph::Node>& '
        s += self._name
        return s

    def getter_proto(self, is_const):
        s = ''
        s_const = 'const ' if is_const else ''
        s += ('::ngraph::Input<%s::ngraph::Node> get_' % s_const)
        s += self._name
        s += ('() %s{ ' % s_const)
        s += ('return input(%d);' % self._idx)
        s += ' }'
        return s

    def get_argument_case(self):
        s = ''
        if self._idx != 0:
            s += 'else '
        s += ('if (name == "%s") { return input(%d); }' %
              (self._name, self._idx))
        return s

    def name(self):
        return self._name


class OpResultWriter():
    def __init__(self, result, idx):
        self._name = result.name
        self._description = result.description
        self._idx = idx

    def getter_proto(self, is_const):
        s = ''
        s_const = 'const ' if is_const else ''
        s += ('::ngraph::Output<%s::ngraph::Node> get_' % s_const)
        s += self._name
        s += ('() %s{ ' % s_const)
        s += ('return output(%d);' % self._idx)
        s += ' }'
        return s

    def get_result_case(self):
        s = ''
        if self._idx != 0:
            s += 'else '
        s += ('if (name == "%s") { return output(%d); }' %
              (self._name, self._idx))
        return s

    def name(self):
        return self._name


class OpAttributeWriter():
    def __init__(self, attribute, idx):
        self._name = attribute.name
        self._type = attribute.type
        self._description = attribute.description
        self._idx = idx

    def ctor_parameter_proto(self):
        return('const %s& %s' % (self._type, self._name))

    def getter_proto(self):
        return ('const %s& get_%s() const { return %s.get(); }'
                % (self._type, self._name, self.member_var_name()))

    def setter_proto(self):
        return ('void set_%s (const %s& %s) { %s.set(%s); }'
                % (self._name, self._type, self._name, self.member_var_name(), self._name))

    def get_attribute_case(self):
        s = ''
        if self._idx != 0:
            s += 'else '
        s += ('if (name == "%s") { return %s; }' %
              (self._name, self.member_var_name()))
        return s

    def ctor_type_check(self):
        return('NGRAPH_CHECK(attributes[%d]->has_type<%s>(), "Attribute %d (name: %s) has incorrect type (%s expected)");'
               % (self._idx, self._type, self._idx, self._name, self._type))

    def ctor_assignment(self):
        return('%s.set(attributes[%d]->as_type<%s>().get());'
               % (self.member_var_name(), self._idx, self._type))

    def member_var_proto(self):
        return ('::ngraph::Attribute<%s> %s;' % (self._type, self.member_var_name()))

    def member_var_name(self):
        return 'm_' + self._name

    def name(self):
        return self._name


class ClassWriter():
    def __init__(self, f):
        self._load_op_json(f)

    def _load_op_json(self, f):
        k = SpecParser().parse_class(json.load(f))

        self._name = k.name
        self._dialect = k.dialect
        self._description = k.description
        self._commutative = k.commutative
        self._has_state = k.has_state
        self._validation_implemented = k.validation_implemented
        self._adjoints_implemented = k.adjoints_implemented
        self._zdte_implemented = k.zdte_implemented
        self._arguments = [OpArgumentWriter(arg,idx) for (idx, arg) in enumerate(k._arguments)]
        self._results = [OpResultWriter(res,idx) for (idx, res) in enumerate(k._results)]
        self._attributes = [OpAttributeWriter(attr,idx) for (idx, attr) in enumerate(k._attributes)]

        f.close()

    def qualified_name(self):
        return ('ngraph::op::gen::%s::%s' % (self._dialect, self._name))

    def gen_copyright_header(self, f):
        f.write("""\
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

""")

    def gen_class_declaration(self, f):
        f.write('namespace ngraph\n')
        f.write('{\n')
        f.write('    namespace op\n')
        f.write('    {\n')
        f.write('        namespace gen\n')
        f.write('        {\n')
        f.write('            namespace %s\n' % self._dialect)
        f.write('            {\n')
        f.write(
            '                class %s;\n' % self._name)
        f.write('            } // namespace %s\n' % self._dialect)
        f.write('        } // namespace gen\n')
        f.write('    } // namespace op\n')
        f.write('} // namespace ngraph\n')
        f.write('\n')

    def gen_class_definition(self, f):
        f.write('// ')
        f.write(self._description)
        f.write('\n')
        f.write('class ::%s final : public ::ngraph::op::util::GenOp\n' %
                self.qualified_name())
        f.write('{')
        f.write('\n')

        f.write('public:\n')

        f.write('    NGRAPH_API static const ::std::string type_name;\n')
        f.write(
            '    const std::string& description() const final override { return type_name; }\n')

        f.write('    class Builder;\n')

        f.write('    static ::std::shared_ptr<::ngraph::Node> build(const ::ngraph::OutputVector& source_outputs, const ::std::vector<const ::ngraph::AttributeBase*>& attributes);\n')

        f.write('    %s() = default;\n' % self._name)

        f.write('    %s(' % self._name)

        f.write(', '.join(argument.ctor_parameter_proto()
                          for argument in self._arguments))

        if len(self._arguments) > 0 and len(self._attributes) > 0:
            f.write(', ')

        f.write(', '.join(attribute.ctor_parameter_proto()
                          for attribute in self._attributes))

        f.write(')\n')
        f.write('        : ::ngraph::op::util::GenOp(::ngraph::OutputVector{')
        f.write(', '.join(argument.name() for argument in self._arguments))
        f.write('})\n')

        for attribute in self._attributes:
            f.write('        , %s(%s)\n' %
                    (attribute.member_var_name(), attribute.name()))

        f.write('    {\n')
        f.write('        constructor_validate_and_infer_types();\n')
        f.write('    }\n')

        f.write('    %s(const ::ngraph::OutputVector& source_outputs, const ::std::vector<const ::ngraph::AttributeBase*>& attributes)\n' % self._name)
        f.write('        : ::ngraph::op::util::GenOp(source_outputs)\n')
        f.write('    {\n')
        f.write('        NGRAPH_CHECK(source_outputs.size() == %d, "Source output count should be %d, not ", source_outputs.size());\n'
                % (len(self._arguments), len(self._arguments)))
        f.write('        NGRAPH_CHECK(attributes.size() == %d, "Attribute count should be %d, not ", attributes.size());\n'
                % (len(self._attributes), len(self._attributes)))

        for attribute in self._attributes:
            f.write('        %s\n' % attribute.ctor_type_check())

        for attribute in self._attributes:
            f.write('        %s\n' % attribute.ctor_assignment())

        f.write('        constructor_validate_and_infer_types();\n')
        f.write('    }\n')

        for argument in self._arguments:
            f.write('    %s\n' % argument.getter_proto(is_const=False))

        for argument in self._arguments:
            f.write('    %s\n' % argument.getter_proto(is_const=True))

        for result in self._results:
            f.write('    %s\n' % result.getter_proto(is_const=False))

        for result in self._results:
            f.write('    %s\n' % result.getter_proto(is_const=True))

        for attribute in self._attributes:
            f.write('    %s\n' % attribute.getter_proto())

        for attribute in self._attributes:
            f.write('    %s\n' % attribute.setter_proto())

        f.write(
            '    ::std::vector<::std::string> get_argument_keys() const final override { return ::std::vector<::std::string>{')
        f.write(', '.join('"%s"' % argument.name()
                          for argument in self._arguments))
        f.write('}; }\n')

        f.write(
            '    ::std::vector<::std::string> get_result_keys() const final override { return ::std::vector<::std::string>{')
        f.write(', '.join('"%s"' % result.name() for result in self._results))
        f.write('}; }\n')

        f.write(
            '    ::std::vector<::std::string> get_attribute_keys() const final override { return ::std::vector<::std::string>{')
        f.write(', '.join('"%s"' % attribute.name()
                          for attribute in self._attributes))
        f.write('}; }\n')

        f.write('    ::ngraph::Input<const ::ngraph::Node> get_argument(const ::std::string& name) const final override\n')
        f.write('    {\n')
        for argument in self._arguments:
            f.write('        %s\n' % argument.get_argument_case())
        if len(self._arguments) > 0:
            f.write('        else\n')
            f.write('        {\n')
            f.write('    ')
        f.write(
            '        NGRAPH_CHECK(false, "get_argument: Invalid argument name ", name);\n')
        if len(self._arguments) > 0:
            f.write('        }\n')
            f.write('    }\n')

        f.write(
            '    ::ngraph::Input<::ngraph::Node> get_argument(const ::std::string& name) final override\n')
        f.write('    {\n')
        for argument in self._arguments:
            f.write('        %s\n' % argument.get_argument_case())
        if len(self._arguments) > 0:
            f.write('        else\n')
            f.write('        {\n')
            f.write('    ')
        f.write(
            '        NGRAPH_CHECK(false, "get_argument: Invalid argument name ", name);\n')
        if len(self._arguments) > 0:
            f.write('        }\n')
            f.write('    }\n')

        f.write('    ::ngraph::Output<const ::ngraph::Node> get_result(const ::std::string& name) const final override\n')
        f.write('    {\n')
        for result in self._results:
            f.write('        %s\n' % result.get_result_case())
        if len(self._results) > 0:
            f.write('        else\n')
            f.write('        {\n')
            f.write('    ')
        f.write(
            '        NGRAPH_CHECK(false, "get_result: Invalid result name ", name);\n')
        if len(self._results) > 0:
            f.write('        }\n')
            f.write('    }\n')

        f.write(
            '    ::ngraph::Output<::ngraph::Node> get_result(const ::std::string& name) final override\n')
        f.write('    {\n')
        for result in self._results:
            f.write('        %s\n' % result.get_result_case())
        if len(self._results) > 0:
            f.write('        else\n')
            f.write('        {\n')
            f.write('    ')
        f.write(
            '        NGRAPH_CHECK(false, "get_result: Invalid result name ", name);\n')
        if len(self._results) > 0:
            f.write('        }\n')
            f.write('    }\n')

        f.write('    const ::ngraph::AttributeBase& get_attribute(const ::std::string& name) const final override\n')
        f.write('    {\n')
        for attribute in self._attributes:
            f.write('        %s\n' % attribute.get_attribute_case())
        if len(self._attributes) > 0:
            f.write('        else\n')
            f.write('        {\n')
            f.write('    ')
        f.write(
            '        NGRAPH_CHECK(false, "get_attribute: Invalid attribute name ", name);\n')
        if len(self._attributes) > 0:
            f.write('        }\n')
            f.write('    }\n')

        f.write('    bool is_commutative() const final override { return %s; }\n' % (
            'true' if self._commutative else 'false'))
        f.write('    bool has_state() const final override { return %s; }\n' % (
            'true' if self._has_state else 'false'))

        f.write('    ::std::shared_ptr<::ngraph::Node> copy_with_new_args\n')
        f.write('        (const ::ngraph::NodeVector& inputs)\n')
        f.write('        const final override\n')
        f.write('    {\n')
        f.write('        NGRAPH_CHECK(inputs.size() == %d, "New argument count should be %d, not ", inputs.size());\n'
                % (len(self._arguments), len(self._arguments)))
        f.write(
            '        ::std::shared_ptr<::ngraph::Node> new_node = ::std::make_shared<%s>(' % self._name)
        f.write(', '.join('inputs[%d]' % idx
                          for (idx, _) in enumerate(self._arguments)))
        if len(self._arguments) > 0 and len(self._attributes) > 0:
            f.write(', ')
        f.write(', '.join(attribute.member_var_name() + ".get()"
                          for attribute in self._attributes))
        f.write(');\n')
        f.write('        // TODO: control deps\n')
        f.write('        return new_node;\n')
        f.write('    }\n')

        if self._zdte_implemented:
            f.write(
                '    ::std::shared_ptr<::ngraph::Node> get_default_value() const final override;\n')

        f.write('protected:\n')

        if self._validation_implemented:
            f.write('    void validate_and_infer_types() final override;\n')

        if self._adjoints_implemented:
            f.write(
                '    void generate_adjoints(::ngraph::autodiff::Adjoints& adjoints,\n')
            f.write(
                '                           const ::ngraph::NodeVector& deltas) final override;\n')

        f.write('private:\n')

        for attribute in self._attributes:
            f.write('    %s\n' % attribute.member_var_proto())

        f.write('    NGRAPH_API static bool s_registered;\n')
        f.write('};\n')

    def gen_builder_class_definition(self, f):
        f.write('class ::%s::Builder final : public ::ngraph::GenOpBuilder\n' %
                self.qualified_name())
        f.write('{')
        f.write('\n')

        f.write('public:\n')
        f.write(
            '    ::std::shared_ptr<::ngraph::Node> build(const ::ngraph::OutputVector& source_outputs,\n')
        f.write('                                            const ::std::vector<const ::ngraph::AttributeBase*>& attributes)\n')
        f.write('               const final override\n')
        f.write('    {\n')
        f.write('        return ::%s::build(source_outputs, attributes);\n'
                % self.qualified_name())
        f.write('    }\n')
        f.write('};\n')

    def gen_hpp_file(self, outdir_name):
        os.makedirs('%s/ngraph/op/gen/%s' %
                    (outdir_name, self._dialect), exist_ok=True)

        f = open('%s/ngraph/op/gen/%s/%s.hpp' % (outdir_name,
                                                 self._dialect, inflection.underscore(self._name)), 'w')
        self.gen_copyright_header(f)

        f.write('#pragma once\n')
        f.write('\n')
        f.write('#include "ngraph/op/util/gen_op.hpp"\n')
        f.write('\n')

        self.gen_class_declaration(f)

        self.gen_class_definition(f)

        self.gen_builder_class_definition(f)

        f.close()

    def gen_function_definitions(self, f):
        f.write('const std::string %s::type_name = "%s.%s";\n' %
                (self.qualified_name(), self._dialect, self._name))
        f.write('\n')

        f.write('::std::shared_ptr<::ngraph::Node> %s::build(const ::ngraph::OutputVector& source_outputs, const ::std::vector<const AttributeBase*>& attributes)\n'
                % self.qualified_name())
        f.write('{\n')
        f.write('    return ::std::make_shared<::%s>(source_outputs, attributes);\n'
                % self.qualified_name())
        f.write('}\n')

    def gen_cpp_file(self, outdir_name):
        os.makedirs('%s/ngraph/op/gen/%s' %
                    (outdir_name, self._dialect), exist_ok=True)

        f = open('%s/ngraph/op/gen/%s/%s.cpp' % (outdir_name,
                                                 self._dialect, inflection.underscore(self._name)), 'w')
        self.gen_copyright_header(f)

        f.write('#include "ngraph/op/gen/%s/%s.hpp"\n' %
                (self._dialect, inflection.underscore(self._name)))
        f.write('\n')

        self.gen_function_definitions(f)

        f.write('bool ::%s::s_registered = ::ngraph::register_gen_op("%s.%s", new ::%s::Builder());\n'
                % (self.qualified_name(), self._dialect, self._name, self.qualified_name()))

        f.close()
