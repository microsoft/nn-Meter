import inspect

class Registry:
    ''' Modified from https://github.com/ultmaster/utilsd/blob/master/utilsd/config/registry.py
    '''
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def items(self):
        return self._module_dict

    def __repr__(self):
        format_str = self._name + f'(name={self._name}, items={self._module_dict})'
        return format_str

    def get(self, key):
        if key in self._module_dict:
            return self._module_dict[key]

    def _register(self, module_class, module_name=None):
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__

        if module_name in self._module_dict:
            self._module_dict.update({module_name: module_class})
        self._module_dict[module_name] = module_class

    def register(self, name, module = None):
        # use it as a normal method: x.register(name="my_class", module=MyClass)
        if module is not None:
            self._register(module_class=module, module_name=name)
            return module

        # use it as a decorator: @x.register(name="my_class")
        def _register(reg_self):
            self._register(
                module_class=reg_self, module_name=name)
            return reg_self

        return _register
