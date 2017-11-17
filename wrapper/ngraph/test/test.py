import wrapper.ngraph.types.clsTraitedType as clsTraitedType
import wrapper.ngraph.ops.clsParameter as clsParameter
import wrapper.ngraph.runtime.clsTensorViewType as clsTensorViewType
import wrapper.ngraph.clsFunction as clsFunction
import wrapper.ngraph.runtime.clsManager as clsManager
import wrapper.ngraph.runtime.clsParameterizedTensorView as clsParameterizedTensorView

element_type = clsTraitedType.TraitedTypeF.element_type()
shape = [2,2]
A = clsParameter.Parameter(element_type, shape)
B = clsParameter.Parameter(element_type, shape)
C = clsParameter.Parameter(element_type, shape)
value_type = clsTensorViewType.TensorViewType(element_type, shape)
parameter_list = [A, B, C]
function = clsFunction.Function((A + B) * C, value_type, parameter_list, 'test')
manager = clsManager.Manager.get('NGVM');
external = manager.compile(function)
backend = manager.allocate_backend()
cf = backend.make_call_frame(external)
a = backend.make_primary_tensor_view(element_type, shape)
b = backend.make_primary_tensor_view(element_type, shape)
c = backend.make_primary_tensor_view(element_type, shape)
result = backend.make_primary_tensor_view(element_type, shape)
cf.call([a, b, c], [result])
