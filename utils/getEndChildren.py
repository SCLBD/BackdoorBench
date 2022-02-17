import sys, logging
sys.path.append('../')
def getEndChildren(model):
    def layerTraverse(module_node):
        if getattr(module_node, 'children') is None:
            return None
        queue = []
        queue.append(module_node)
        endChildrenList = []
        while len(queue) > 0:
            tmp = queue.pop(0)
            if len(list(tmp.children())) == 0:
                endChildrenList.append(tmp)
            else:
                queue += tmp.children()
        return endChildrenList
    return layerTraverse(model)
if __name__ == '__main__':

    from torchvision.models import resnet18

    a = resnet18()

    print(a)
    from pprint import pprint
    pprint(getEndChildren(a))