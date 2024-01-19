import sys
from collections import defaultdict
from inspect import Parameter, signature
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, DefaultDict, Generator, Sequence, TypedDict, TypeVar, cast

from torch import device as Device, dtype as DType
from torch.nn.modules.module import Module as TorchModule

from imaginairy.vendored.refiners.fluxion.context import Context, ContextProvider
from imaginairy.vendored.refiners.fluxion.utils import load_from_safetensors

if TYPE_CHECKING:
    from imaginairy.vendored.refiners.fluxion.layers.chain import Chain

T = TypeVar("T", bound="Module")
TContextModule = TypeVar("TContextModule", bound="ContextModule")
BasicType = str | float | int | bool


class Module(TorchModule):
    _parameters: dict[str, Any]
    _buffers: dict[str, Any]
    _tag: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, *kwargs)  # type: ignore[reportUnknownMemberType]

    def __getattr__(self, name: str) -> Any:
        return super().__getattr__(name=name)

    def __setattr__(self, name: str, value: Any) -> None:
        return super().__setattr__(name=name, value=value)

    def load_from_safetensors(self, tensors_path: str | Path, strict: bool = True) -> "Module":
        state_dict = load_from_safetensors(tensors_path)
        self.load_state_dict(state_dict, strict=strict)
        return self

    def named_modules(self, *args: Any, **kwargs: Any) -> "Generator[tuple[str, Module], None, None]":  # type: ignore
        return super().named_modules(*args)  # type: ignore

    def to(self: T, device: Device | str | None = None, dtype: DType | None = None) -> T:  # type: ignore
        return super().to(device=device, dtype=dtype)  # type: ignore

    def __str__(self) -> str:
        basic_attributes_str = ", ".join(
            f"{key}={value}" for key, value in self.basic_attributes(init_attrs_only=True).items()
        )
        result = f"{self.__class__.__name__}({basic_attributes_str})"
        return result

    def __repr__(self) -> str:
        tree = ModuleTree(module=self)
        return repr(tree)

    def pretty_print(self, depth: int = -1) -> None:
        tree = ModuleTree(module=self)
        print(tree._generate_tree_repr(tree.root, is_root=True, depth=depth))  # type: ignore[reportPrivateUsage]

    def basic_attributes(self, init_attrs_only: bool = False) -> dict[str, BasicType]:
        """Return a dictionary of basic attributes of the module.

        Basic attributes are public attributes made of basic types (int, float, str, bool) or a sequence of basic types.
        """
        sig = signature(obj=self.__init__)
        init_params = set(sig.parameters.keys()) - {"self"}
        default_values = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}

        def is_basic_attribute(key: str, value: Any) -> bool:
            if key.startswith("_"):
                return False

            if isinstance(value, BasicType):
                return True

            if isinstance(value, Sequence) and all(isinstance(y, BasicType) for y in cast(Sequence[Any], value)):
                return True

            return False

        return {
            key: str(object=value)
            for key, value in self.__dict__.items()
            if is_basic_attribute(key=key, value=value)
            and (not init_attrs_only or (key in init_params and value != default_values.get(key)))
        }

    def _show_only_tag(self) -> bool:
        """Whether to show only the tag when printing the module.

        This is useful to distinguish between Chain subclasses that override their forward from one another.
        """
        return False


class ContextModule(Module):
    # we store parent into a one element list to avoid pytorch thinking it's a submodule
    _parent: "list[Chain]"
    _can_refresh_parent: bool = True  # see usage in Adapter and Chain

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, *kwargs)
        self._parent = []

    @property
    def parent(self) -> "Chain | None":
        return self._parent[0] if self._parent else None

    @property
    def ensure_parent(self) -> "Chain":
        assert self._parent, "module is not bound to a Chain"
        return self._parent[0]

    def _set_parent(self, parent: "Chain | None") -> None:
        if not self._can_refresh_parent:
            return
        if parent is None:
            self._parent = []
            return
        # Always insert the module in the Chain first to avoid inconsistencies.
        assert self in iter(parent), f"{self} not in {parent}"
        self._parent = [parent]

    @property
    def provider(self) -> ContextProvider:
        return self.ensure_parent.provider

    def get_parents(self) -> "list[Chain]":
        return self._parent + self._parent[0].get_parents() if self._parent else []

    def use_context(self, context_name: str) -> Context:
        """Retrieve the context object from the module's context provider."""
        context = self.provider.get_context(context_name)
        assert context is not None, f"Context {context_name} not found."
        return context

    def structural_copy(self: TContextModule) -> TContextModule:
        clone = object.__new__(self.__class__)

        not_torch_attributes = [
            key
            for key, value in self.__dict__.items()
            if not key.startswith("_")
            and isinstance(sys.modules.get(type(value).__module__), ModuleType)
            and "torch" not in sys.modules[type(value).__module__].__name__
        ]

        for k in not_torch_attributes:
            setattr(clone, k, getattr(self, k))

        ContextModule.__init__(self=clone)

        return clone


class WeightedModule(Module):
    @property
    def device(self) -> Device:
        return self.weight.device

    @property
    def dtype(self) -> DType:
        return self.weight.dtype

    def __str__(self) -> str:
        return f"{super().__str__().removesuffix(')')}, device={self.device}, dtype={str(self.dtype).removeprefix('torch.')})"


class TreeNode(TypedDict):
    value: str
    class_name: str
    children: list["TreeNode"]


class ModuleTree:
    def __init__(self, module: Module) -> None:
        self.root: TreeNode = self._module_to_tree(module=module)
        self._fold_successive_identical(node=self.root)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root['value']})"

    def __repr__(self) -> str:
        return self._generate_tree_repr(self.root, is_root=True, depth=7)

    def __iter__(self) -> Generator[TreeNode, None, None]:
        for child in self.root["children"]:
            yield child

    @classmethod
    def shorten_tree_repr(cls, tree_repr: str, /, line_index: int = 0, max_lines: int = 20) -> str:
        """Shorten the tree representation to a given number of lines around a given line index."""
        lines = tree_repr.split(sep="\n")
        start_idx = max(0, line_index - max_lines // 2)
        end_idx = min(len(lines), line_index + max_lines // 2 + 1)
        return "\n".join(lines[start_idx:end_idx])

    def _generate_tree_repr(
        self, node: TreeNode, /, *, prefix: str = "", is_last: bool = True, is_root: bool = True, depth: int = -1
    ) -> str:
        if depth == 0 and node["children"]:
            return f"{prefix}{'└── ' if is_last else '├── '}{node['value']} ..."

        if depth > 0:
            depth -= 1

        tree_icon: str = "" if is_root else ("└── " if is_last else "├── ")
        counts: DefaultDict[str, int] = defaultdict(int)

        for child in node["children"]:
            counts[child["class_name"]] += 1

        instance_counts: DefaultDict[str, int] = defaultdict(int)
        lines = [f"{prefix}{tree_icon}{node['value']}"]
        new_prefix: str = "    " if is_last else "│   "

        for i, child in enumerate(iterable=node["children"]):
            instance_counts[child["class_name"]] += 1

            if counts[child["class_name"]] > 1:
                child_value = f"{child['value']} #{instance_counts[child['class_name']]}"
            else:
                child_value = child["value"]

            child_str = self._generate_tree_repr(
                {"value": child_value, "class_name": child["class_name"], "children": child["children"]},
                prefix=prefix + new_prefix,
                is_last=i == len(node["children"]) - 1,
                is_root=False,
                depth=depth,
            )

            if child_str:
                lines.append(child_str)

        return "\n".join(lines)

    def _module_to_tree(self, module: Module) -> TreeNode:
        match (module._tag, module._show_only_tag()):  # pyright: ignore[reportPrivateUsage]
            case ("", False):
                value = str(module)
            case (_, True):
                value = f"({module._tag})"  # pyright: ignore[reportPrivateUsage]
            case (_, False):
                value = f"({module._tag}) {module}"  # pyright: ignore[reportPrivateUsage]

        class_name = module.__class__.__name__

        node: TreeNode = {"value": value, "class_name": class_name, "children": []}
        for child in module.children():
            node["children"].append(self._module_to_tree(module=child))  # type: ignore
        return node

    def _fold_successive_identical(self, node: TreeNode) -> None:
        i = 0
        while i < len(node["children"]):
            j = i
            while j < len(node["children"]) and node["children"][i] == node["children"][j]:
                j += 1
            count = j - i
            if count > 1:
                node["children"][i]["value"] += f" (x{count})"
                del node["children"][i + 1 : j]
            self._fold_successive_identical(node=node["children"][i])
            i += 1
