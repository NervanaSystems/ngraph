import clsTraitedType
import clsParameter
import clsTensorViewType
import clsFunction


element_type = clsTraitedType.TraitedTypeF.element_type()
shape = [2,2]
a = clsParameter.clsParameter(element_type, shape)
b = clsParameter.clsParameter(element_type, shape)
c = clsParameter.clsParameter(element_type, shape)
value_type = clsTensorViewType.clsTensorViewType(element_type, shape)
parameter_list = [a, b, c]
function = clsFunction.clsFunction((a + b)*c, value_type, parameter_list, 'test')
print(function.get_result_type())
