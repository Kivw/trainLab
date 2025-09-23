"""
Author: Kanna
Date: 2025-09-23
Version: 0.1
License: MIT
"""

import inspect
import sys
from typing import Any, Dict, List, Optional, Type, Union


class Registry:
    """通用注册表，可注册类或函数，支持多层 scope 和自动构建实例."""

    def __init__(self,
                 name: str,
                 parent: Optional['Registry'] = None,
                 scope: Optional[str] = None,
                 locations: List[str] = []):
        self._name = name
        self._module_dict: Dict[str, Any] = {} # 存储注册的模块
        self._children: Dict[str, 'Registry'] = {} # 子注册表
        self._locations = locations
        self._imported = False

        # scope
        self._scope = scope if scope else self.infer_scope()

        # parent
        self.parent: Optional['Registry'] = None
        if parent:
            parent._add_child(self)
            self.parent = parent

    @staticmethod
    def infer_scope() -> str:
        """尝试自动推断 scope"""
        frame = sys._getframe(2) # 上上层调用的帧对象
        module = inspect.getmodule(frame) # 根据帧对象找到它所属的 Python 模块
        if module:
            return module.__name__.split('.')[0]
        return 'default_scope'

    def _add_child(self, registry: 'Registry'):
        assert registry.scope not in self._children, f"{registry.scope} exists"
        self._children[registry.scope] = registry

    def _register_module(self, module: Any, name: Optional[Union[str, List[str]]] = None, force: bool = False):
        """注册模块，可是 class、function 或 config dict"""
        if name is None:
            name = getattr(module, '__name__', str(module))
        if isinstance(name, str):
            name = [name]
        for n in name:
            if not force and n in self._module_dict:
                raise KeyError(f"{n} already registered in {self._name}")
            self._module_dict[n] = module

    def register_module(self, name: Optional[Union[str, List[str]]] = None, force: bool = False, module: Optional[Any] = None):
        """注册模块装饰器"""
        if module is not None:
            self._register_module(module=module, name=name, force=force)
            return module

        def _register(cls_or_fn):
            self._register_module(module=cls_or_fn, name=name, force=force)
            return cls_or_fn
        return _register

    def get(self, key: str) -> Any:
        """根据名字获取模块"""
        if key in self._module_dict:
            return self._module_dict[key]
        if self.parent:
            return self.parent.get(key)
        raise KeyError(f"{key} not found in registry {self._name}")

    def build(self, cfg: dict, *args, **kwargs) -> Any:
        """构建实例, 适用于 class 或 function"""
        obj_type = cfg.get('type')
        if obj_type is None:
            raise ValueError("cfg must have 'type' key")
        obj_cls = self.get(obj_type)
        if callable(obj_cls):
            params = cfg.copy()
            params.pop('type')
            return obj_cls(*args, **params, **kwargs)
        return obj_cls

    def exists(self, name: str) -> bool:
        """检查模块是否在当前注册表或父注册表中存在"""
        if name in self._module_dict:
            return True
        if self.parent:
            return self.parent.exists(name)
        return False

    def __repr__(self):
        lines = [f"Registry: {self._name}", "-" * 40]
        lines.append(f"{'Name':20} | {'Object'}")
        lines.append("-" * 40)
        for k, v in self._module_dict.items():
            obj_str = getattr(v, "__name__", str(v))
            lines.append(f"{k:20} | {obj_str}")
        lines.append("-" * 40)
        return "\n".join(lines)

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict
