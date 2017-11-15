import clsTraitedType
import clsParameter
import clsTensorViewType
import clsFunction
import clsManager
import clsParameterizedTensorView
import numpy as np

element_type = clsTraitedType.TraitedTypeF.element_type()
shape = [2,2]
A = clsParameter.clsParameter(element_type, shape)
B = clsParameter.clsParameter(element_type, shape)
C = clsParameter.clsParameter(element_type, shape)
value_type = clsTensorViewType.clsTensorViewType(element_type, shape)
parameter_list = [A, B, C]
function = clsFunction.clsFunction((A + B) * C, value_type, parameter_list, 'test')
manager = clsManager.clsManager.get('NGVM');
external = manager.compile(function)
backend = manager.allocate_backend()
cf = backend.make_call_frame(external)

a = backend.make_primary_tensor_view(element_type, shape)
b = backend.make_primary_tensor_view(element_type, shape)
c = backend.make_primary_tensor_view(element_type, shape)

a.write(np.array[1, 2, 3, 4], 0, 4)
