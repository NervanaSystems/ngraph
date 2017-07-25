import ngraph as ng
C, H, N = ng.make_axis(5), ng.make_axis(3), ng.make_axis(7)
a = ng.Axes(axes=[C, H, N])
b = ng.Axes(axes=[[C, H], N])
print('a={}'.format(a))
print('b={}'.format(b))
print('a[0]={}'.format(a[0]))
print('a[1]={}'.format(a[1]))
print('a[2]={}'.format(a[2]))
print('b[0]={}'.format(b[0]))
print('b[1]={}'.format(b[1]))

print('as_nested_list(a)={}'.format(ng.Axes.as_nested_list(a)))
print('as_flattened_list(a)={}'.format(ng.Axes.as_flattened_list(a)))

print('as_nested_list(b)={}'.format(ng.Axes.as_nested_list(b)))
print('as_flattened_list(b)={}'.format(ng.Axes.as_flattened_list(b)))
