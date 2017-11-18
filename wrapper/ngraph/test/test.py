import numpy as np

import wrapper.ngraph.clsUtil as clsUtil
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

a.write(clsUtil.numpy_to_c(np.array([1,2,3,4], dtype=np.float32)), 0, 16)
b.write(clsUtil.numpy_to_c(np.array([5,6,7,8], dtype=np.float32)), 0, 16)
c.write(clsUtil.numpy_to_c(np.array([9,10,11,12], dtype=np.float32)), 0, 16)

result_arr = np.array([0, 0, 0, 0], dtype=np.float32)
result.write(clsUtil.numpy_to_c(result_arr), 0, 16)

cf.call([a, b, c], [result])

result.read(clsUtil.numpy_to_c(result_arr), 0, 16)
print result_arr
