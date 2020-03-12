from _pyngraph import NodeFactory as _NodeFactory


class NodeFactory(object):

    def create(self, op_type_name, arguments, attributes=None):  # type: (str, List[Node], Optional[Dict[str, Any]]) -> Node
        if attributes is None:
            attributes = {}
        factory = _NodeFactory('opset1')
        node = factory.create(op_type_name, arguments, attributes)
        return node
