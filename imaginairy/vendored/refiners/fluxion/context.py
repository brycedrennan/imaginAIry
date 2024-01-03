from typing import Any

from torch import Tensor

Context = dict[str, Any]
Contexts = dict[str, Context]


class ContextProvider:
    def __init__(self) -> None:
        self.contexts: Contexts = {}

    def set_context(self, key: str, value: Context) -> None:
        self.contexts[key] = value

    def get_context(self, key: str) -> Any:
        return self.contexts.get(key)

    def update_contexts(self, new_contexts: Contexts) -> None:
        for key, value in new_contexts.items():
            if key not in self.contexts:
                self.contexts[key] = value
            else:
                self.contexts[key].update(value)

    @staticmethod
    def create(contexts: Contexts) -> "ContextProvider":
        provider = ContextProvider()
        provider.update_contexts(contexts)
        return provider

    def __add__(self, other: "ContextProvider") -> "ContextProvider":
        self.contexts.update(other.contexts)
        return self

    def __lshift__(self, other: "ContextProvider") -> "ContextProvider":
        other.contexts.update(self.contexts)
        return other

    def __bool__(self) -> bool:
        return bool(self.contexts)

    def _get_repr_for_value(self, value: Any) -> str:
        if isinstance(value, Tensor):
            return f"Tensor(shape={value.shape}, dtype={value.dtype}, device={value.device})"
        return repr(value)

    def _get_repr_for_dict(self, context_dict: Context) -> dict[str, str]:
        return {key: self._get_repr_for_value(value) for key, value in context_dict.items()}

    def __repr__(self) -> str:
        contexts_repr = {key: self._get_repr_for_dict(value) for key, value in self.contexts.items()}
        return f"{self.__class__.__name__}(contexts={contexts_repr})"
