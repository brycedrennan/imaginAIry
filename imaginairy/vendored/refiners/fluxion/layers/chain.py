import inspect
import re
import sys
import traceback
from collections import defaultdict
from typing import Any, Callable, Iterable, Iterator, TypeVar, cast, overload

import torch
from torch import Tensor, cat, device as Device, dtype as DType

from imaginairy.vendored.refiners.fluxion.context import ContextProvider, Contexts
from imaginairy.vendored.refiners.fluxion.layers.module import ContextModule, Module, ModuleTree, WeightedModule
from imaginairy.vendored.refiners.fluxion.utils import summarize_tensor

T = TypeVar("T", bound=Module)
TChain = TypeVar("TChain", bound="Chain")  # because Self (PEP 673) is not in 3.10


class Lambda(Module):
    """Lambda is a wrapper around a callable object that allows it to be used as a PyTorch module."""

    def __init__(self, func: Callable[..., Any]) -> None:
        super().__init__()
        self.func = func

    def forward(self, *args: Any) -> Any:
        return self.func(*args)

    def __str__(self) -> str:
        func_name = getattr(self.func, "__name__", "partial_function")
        return f"Lambda({func_name}{str(inspect.signature(self.func))})"


def generate_unique_names(
    modules: tuple[Module, ...],
) -> dict[str, Module]:
    class_counts: dict[str, int] = {}
    unique_names: list[tuple[str, Module]] = []
    for module in modules:
        class_name = module.__class__.__name__
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    name_counter: dict[str, int] = {}
    for module in modules:
        class_name = module.__class__.__name__
        name_counter[class_name] = name_counter.get(class_name, 0) + 1
        unique_name = f"{class_name}_{name_counter[class_name]}" if class_counts[class_name] > 1 else class_name
        unique_names.append((unique_name, module))
    return dict(unique_names)


class UseContext(ContextModule):
    def __init__(self, context: str, key: str) -> None:
        super().__init__()
        self.context = context
        self.key = key
        self.func: Callable[[Any], Any] = lambda x: x

    def __call__(self, *args: Any) -> Any:
        context = self.use_context(self.context)
        assert context, f"context {self.context} is unset"
        value = context.get(self.key)
        assert value is not None, f"context entry {self.context}.{self.key} is unset"
        return self.func(value)

    def __repr__(self):
        return f"{self.__class__.__name__}(context={repr(self.context)}, key={repr(self.key)})"

    def compose(self, func: Callable[[Any], Any]) -> "UseContext":
        self.func = func
        return self


class SetContext(ContextModule):
    """A Module that sets a context value when executed.

    The context need to pre exist in the context provider.
    #TODO Is there a way to create the context if it doesn't exist?
    """

    def __init__(self, context: str, key: str, callback: Callable[[Any, Any], Any] | None = None) -> None:
        super().__init__()
        self.context = context
        self.key = key
        self.callback = callback

    def __call__(self, x: Tensor) -> Tensor:
        if context := self.use_context(self.context):
            if not self.callback:
                context.update({self.key: x})
            else:
                self.callback(context[self.key], x)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(context={repr(self.context)}, key={repr(self.key)})"


class ReturnException(Exception):
    """Exception raised when a Return module is encountered."""

    def __init__(self, value: Tensor):
        self.value = value


class Return(Module):
    """A Module that stops the execution of a Chain when encountered."""

    def forward(self, x: Tensor):
        raise ReturnException(x)


def structural_copy(m: T) -> T:
    return m.structural_copy() if isinstance(m, ContextModule) else m


class ChainError(RuntimeError):
    """Exception raised when an error occurs during the execution of a Chain."""

    def __init__(self, message: str, /) -> None:
        super().__init__(message)


class Chain(ContextModule):
    _modules: dict[str, Module]
    _provider: ContextProvider
    _tag = "CHAIN"

    def __init__(self, *args: Module | Iterable[Module]) -> None:
        super().__init__()
        self._provider = ContextProvider()
        modules = cast(
            tuple[Module],
            (
                tuple(args[0])
                if len(args) == 1 and isinstance(args[0], Iterable) and not isinstance(args[0], Chain)
                else tuple(args)
            ),
        )

        for module in modules:
            # Violating this would mean a ContextModule ends up in two chains,
            # with a single one correctly set as its parent.
            assert (
                (not isinstance(module, ContextModule))
                or (not module._can_refresh_parent)
                or (module.parent is None)
                or (module.parent == self)
            ), f"{module.__class__.__name__} already has parent {module.parent.__class__.__name__}"

        self._regenerate_keys(modules)
        self._reset_context()

        for module in self:
            if isinstance(module, ContextModule) and module.parent != self:
                module._set_parent(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, torch.nn.Module):
            raise ValueError(
                "Chain does not support setting modules by attribute. Instead, use a mutation method like `append` or"
                " wrap it within a single element list to prevent pytorch from registering it as a submodule."
            )
        super().__setattr__(name, value)

    @property
    def provider(self) -> ContextProvider:
        return self._provider

    def init_context(self) -> Contexts:
        return {}

    def _register_provider(self, context: Contexts | None = None) -> None:
        if context:
            self._provider.update_contexts(context)

        for module in self:
            if isinstance(module, Chain):
                module._register_provider(context=self._provider.contexts)

    def _reset_context(self) -> None:
        self._register_provider(self.init_context())

    def set_context(self, context: str, value: Any) -> None:
        self._provider.set_context(context, value)
        self._register_provider()

    def _show_error_in_tree(self, name: str, /, max_lines: int = 20) -> str:
        tree = ModuleTree(module=self)
        classname_counter: dict[str, int] = defaultdict(int)
        first_ancestor = self.get_parents()[-1] if self.get_parents() else self

        def find_state_dict_key(module: Module, /) -> str | None:
            for key, layer in module.named_modules():
                if layer == self:
                    return ".".join((key, name))
            return None

        for child in tree:
            classname, count = name.rsplit(sep="_", maxsplit=1) if "_" in name else (name, "1")
            if child["class_name"] == classname:
                classname_counter[classname] += 1
                if classname_counter[classname] == int(count):
                    state_dict_key = find_state_dict_key(first_ancestor)
                    child["value"] = f">>> {child['value']} | {state_dict_key}"
                    break

        tree_repr = tree._generate_tree_repr(tree.root, depth=3)  # type: ignore[reportPrivateUsage]

        lines = tree_repr.split(sep="\n")
        error_line_idx = next((idx for idx, line in enumerate(iterable=lines) if line.startswith(">>>")), 0)

        return ModuleTree.shorten_tree_repr(tree_repr, line_index=error_line_idx, max_lines=max_lines)

    @staticmethod
    def _pretty_print_args(*args: Any) -> str:
        """
        Flatten nested tuples and print tensors with their shape and other informations.
        """

        def _flatten_tuple(t: Tensor | tuple[Any, ...], /) -> list[Any]:
            if isinstance(t, tuple):
                return [item for subtuple in t for item in _flatten_tuple(subtuple)]
            else:
                return [t]

        flat_args = _flatten_tuple(args)

        return "\n".join(
            [
                f"{idx}: {summarize_tensor(arg) if isinstance(arg, Tensor) else arg}"
                for idx, arg in enumerate(iterable=flat_args)
            ]
        )

    def _filter_traceback(self, *frames: traceback.FrameSummary) -> list[traceback.FrameSummary]:
        patterns_to_exclude = [
            (r"torch/nn/modules/", r"^_call_impl$"),
            (r"torch/nn/functional\.py", r""),
            (r"refiners/fluxion/layers/", r"^_call_layer$"),
            (r"refiners/fluxion/layers/", r"^forward$"),
            (r"refiners/fluxion/layers/chain\.py", r""),
            (r"", r"^_"),
        ]

        def should_exclude(frame: traceback.FrameSummary, /) -> bool:
            for filename_pattern, name_pattern in patterns_to_exclude:
                if re.search(pattern=filename_pattern, string=frame.filename) and re.search(
                    pattern=name_pattern, string=frame.name
                ):
                    return True
            return False

        return [frame for frame in frames if not should_exclude(frame)]

    def _call_layer(self, layer: Module, name: str, /, *args: Any) -> Any:
        try:
            return layer(*args)
        except Exception as e:
            exc_type, _, exc_traceback = sys.exc_info()
            assert exc_type
            tb_list = traceback.extract_tb(tb=exc_traceback)
            filtered_tb_list = self._filter_traceback(*tb_list)
            formatted_tb = "".join(traceback.format_list(extracted_list=filtered_tb_list))
            pretty_args = Chain._pretty_print_args(args)
            error_tree = self._show_error_in_tree(name)

            exception_str = re.sub(pattern=r"\n\s*\n", repl="\n", string=str(object=e))
            message = f"{formatted_tb}\n{exception_str}\n---------------\n{error_tree}\n{pretty_args}"
            if "Error" not in exception_str:
                message = f"{exc_type.__name__}:\n {message}"

            raise ChainError(message) from None

    def forward(self, *args: Any) -> Any:
        result: tuple[Any] | Any = None
        intermediate_args: tuple[Any, ...] = args
        for name, layer in self._modules.items():
            result = self._call_layer(layer, name, *intermediate_args)
            intermediate_args = (result,) if not isinstance(result, tuple) else result

        self._reset_context()
        return result

    def _regenerate_keys(self, modules: Iterable[Module]) -> None:
        self._modules = generate_unique_names(tuple(modules))  # type: ignore

    def __add__(self, other: "Chain | Module | list[Module]") -> "Chain":
        if isinstance(other, Module):
            other = Chain(other)
        if isinstance(other, list):
            other = Chain(*other)
        return Chain(*self, *other)

    @overload
    def __getitem__(self, key: int) -> Module:
        ...

    @overload
    def __getitem__(self, key: str) -> Module:
        ...

    @overload
    def __getitem__(self, key: slice) -> "Chain":
        ...

    def __getitem__(self, key: int | str | slice) -> Module:
        if isinstance(key, slice):
            copy = self.structural_copy()
            copy._regenerate_keys(modules=list(copy)[key])
            return copy
        elif isinstance(key, str):
            return self._modules[key]
        else:
            return list(self)[key]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    @property
    def device(self) -> Device | None:
        wm = self.find(WeightedModule)
        return None if wm is None else wm.device

    @property
    def dtype(self) -> DType | None:
        wm = self.find(WeightedModule)
        return None if wm is None else wm.dtype

    def _walk(
        self, predicate: Callable[[Module, "Chain"], bool] | None = None, recurse: bool = False
    ) -> Iterator[tuple[Module, "Chain"]]:
        if predicate is None:
            predicate = lambda _m, _p: True
        for module in self:
            try:
                p = predicate(module, self)
            except StopIteration:
                continue
            if p:
                yield (module, self)
                if not recurse:
                    continue
            if isinstance(module, Chain):
                yield from module.walk(predicate, recurse)

    @overload
    def walk(
        self, predicate: Callable[[Module, "Chain"], bool] | None = None, recurse: bool = False
    ) -> Iterator[tuple[Module, "Chain"]]:
        ...

    @overload
    def walk(self, predicate: type[T], recurse: bool = False) -> Iterator[tuple[T, "Chain"]]:
        ...

    def walk(
        self, predicate: type[T] | Callable[[Module, "Chain"], bool] | None = None, recurse: bool = False
    ) -> Iterator[tuple[T, "Chain"]] | Iterator[tuple[Module, "Chain"]]:
        if isinstance(predicate, type):
            return self._walk(lambda m, _: isinstance(m, predicate), recurse)
        else:
            return self._walk(predicate, recurse)

    def layers(self, layer_type: type[T], recurse: bool = False) -> Iterator[T]:
        for module, _ in self.walk(layer_type, recurse):
            yield module

    def find(self, layer_type: type[T]) -> T | None:
        return next(self.layers(layer_type=layer_type), None)

    def ensure_find(self, layer_type: type[T]) -> T:
        r = self.find(layer_type)
        assert r is not None, f"could not find {layer_type} in {self}"
        return r

    def find_parent(self, module: Module) -> "Chain | None":
        if module in self:  # avoid DFS-crawling the whole tree
            return self
        for _, parent in self.walk(lambda m, _: m == module):
            return parent
        return None

    def ensure_find_parent(self, module: Module) -> "Chain":
        r = self.find_parent(module)
        assert r is not None, f"could not find {module} in {self}"
        return r

    def insert(self, index: int, module: Module) -> None:
        if index < 0:
            index = max(0, len(self._modules) + index + 1)
        modules = list(self)
        modules.insert(index, module)
        self._regenerate_keys(modules)
        if isinstance(module, ContextModule):
            module._set_parent(self)
        self._register_provider()

    def insert_before_type(self, module_type: type[Module], new_module: Module) -> None:
        for i, module in enumerate(self):
            if isinstance(module, module_type):
                self.insert(i, new_module)
                return
        raise ValueError(f"No module of type {module_type.__name__} found in the chain.")

    def insert_after_type(self, module_type: type[Module], new_module: Module) -> None:
        for i, module in enumerate(self):
            if isinstance(module, module_type):
                self.insert(i + 1, new_module)
                return
        raise ValueError(f"No module of type {module_type.__name__} found in the chain.")

    def append(self, module: Module) -> None:
        self.insert(-1, module)

    def pop(self, index: int = -1) -> Module | tuple[Module]:
        modules = list(self)
        if index < 0:
            index = len(modules) + index
        if index < 0 or index >= len(modules):
            raise IndexError("Index out of range.")
        removed_module = modules.pop(index)
        if isinstance(removed_module, ContextModule):
            removed_module._set_parent(None)
        self._regenerate_keys(modules)
        return removed_module

    def remove(self, module: Module) -> None:
        """Remove a module from the chain."""
        modules = list(self)
        try:
            modules.remove(module)
        except ValueError:
            raise ValueError(f"{module} is not in {self}")
        self._regenerate_keys(modules)
        if isinstance(module, ContextModule):
            module._set_parent(None)

    def replace(
        self,
        old_module: Module,
        new_module: Module,
        old_module_parent: "Chain | None" = None,
    ) -> None:
        """Replace a module in the chain with a new module."""
        modules = list(self)
        try:
            modules[modules.index(old_module)] = new_module
        except ValueError:
            raise ValueError(f"{old_module} is not in {self}")
        self._regenerate_keys(modules)
        if isinstance(new_module, ContextModule):
            new_module._set_parent(self)
        if isinstance(old_module, ContextModule):
            old_module._set_parent(old_module_parent)

    def structural_copy(self: TChain) -> TChain:
        """Copy the structure of the Chain tree.

        This method returns a recursive copy of the Chain tree where all inner nodes
        (instances of Chain and its subclasses) are duplicated and all leaves
        (regular Modules) are not.

        Such copies can be adapted without disrupting the base model, but do not
        require extra GPU memory since the weights are in the leaves and hence not copied.
        """
        if hasattr(self, "_pre_structural_copy"):
            self._pre_structural_copy()

        modules = [structural_copy(m) for m in self]
        clone = super().structural_copy()
        clone._provider = ContextProvider.create(clone.init_context())

        for module in modules:
            clone.append(module=module)

        if hasattr(clone, "_post_structural_copy"):
            clone._post_structural_copy(self)

        return clone

    def _show_only_tag(self) -> bool:
        return self.__class__ == Chain


class Parallel(Chain):
    _tag = "PAR"

    def forward(self, *args: Any) -> tuple[Tensor, ...]:
        return tuple([self._call_layer(module, name, *args) for name, module in self._modules.items()])

    def _show_only_tag(self) -> bool:
        return self.__class__ == Parallel


class Distribute(Chain):
    _tag = "DISTR"

    def forward(self, *args: Any) -> tuple[Tensor, ...]:
        n, m = len(args), len(self._modules)
        assert n == m, f"Number of positional arguments ({n}) must match number of sub-modules ({m})."
        return tuple([self._call_layer(module, name, arg) for arg, (name, module) in zip(args, self._modules.items())])

    def _show_only_tag(self) -> bool:
        return self.__class__ == Distribute


class Passthrough(Chain):
    _tag = "PASS"

    def forward(self, *inputs: Any) -> Any:
        super().forward(*inputs)
        return inputs

    def _show_only_tag(self) -> bool:
        return self.__class__ == Passthrough


class Sum(Chain):
    _tag = "SUM"

    def forward(self, *inputs: Any) -> Any:
        output = None
        for layer in self:
            layer_output: Any = layer(*inputs)
            if isinstance(layer_output, tuple):
                layer_output = sum(layer_output)  # type: ignore
            output = layer_output if output is None else output + layer_output
        return output

    def _show_only_tag(self) -> bool:
        return self.__class__ == Sum


class Residual(Chain):
    _tag = "RES"

    def forward(self, *inputs: Any) -> Any:
        assert len(inputs) == 1, "Residual connection can only be used with a single input."
        return super().forward(*inputs) + inputs[0]


class Breakpoint(ContextModule):
    def __init__(self, vscode: bool = True):
        super().__init__()
        self.vscode = vscode

    def forward(self, *args: Any):
        if self.vscode:
            import debugpy  # type: ignore

            debugpy.breakpoint()  # type: ignore
        else:
            breakpoint()
        return args[0] if len(args) == 1 else args


class Concatenate(Chain):
    _tag = "CAT"

    def __init__(self, *modules: Module, dim: int = 0) -> None:
        super().__init__(*modules)
        self.dim = dim

    def forward(self, *args: Any) -> Tensor:
        outputs = [module(*args) for module in self]
        return cat([output for output in outputs if output is not None], dim=self.dim)

    def _show_only_tag(self) -> bool:
        return self.__class__ == Concatenate


class Matmul(Chain):
    _tag = "MATMUL"

    def __init__(self, input: Module, other: Module) -> None:
        super().__init__(
            input,
            other,
        )

    def forward(self, *args: Tensor) -> Tensor:
        return torch.matmul(input=self[0](*args), other=self[1](*args))
