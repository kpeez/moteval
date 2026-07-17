"""Named registries so new datasets plug in by name."""

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def add(obj: T) -> T:
            if name in self._entries:
                raise ValueError(f"{self.kind} {name!r} already registered")
            self._entries[name] = obj
            return obj

        return add

    def get(self, name: str) -> T:
        if name not in self._entries:
            known = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {known}")
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)
