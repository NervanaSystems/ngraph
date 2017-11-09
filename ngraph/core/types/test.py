import TraitedType
import clsParameter
import clsTensorViewType
import clsFunction


a = TraitedType.TraitedTypeF.element_type()
shape = [2,2]
p = clsParameter.clsParameter(a, shape)
t = clsTensorViewType.clsTensorViewType(a, shape)
p_list = [p]
f = clsFunction.clsFunction(p, t, p_list, "test")
print(f.get_result_type())
