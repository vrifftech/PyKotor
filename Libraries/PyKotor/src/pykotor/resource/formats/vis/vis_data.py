"""This module handles classes relating to editing VIS files."""

from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any

from pykotor.resource.type import ResourceType

if TYPE_CHECKING:
    from collections.abc import Generator


class VIS:
    """A room graph with reciprocal visibility, as used by the engine."""

    BINARY_TYPE = ResourceType.VIS

    def __init__(
        self,
    ):
        self._rooms: set[str] = set()
        self._visibility: dict[str, set[str]] = {}

    def __iter__(
        self,
    ) -> Generator[tuple[str, set[str]], Any, None]:
        for observer, observed in self._visibility.items():
            yield observer, deepcopy(observed)

    def all_rooms(
        self,
    ) -> set[str]:
        """Returns a copy of the set of rooms.

        Args:
        ----
            self: The class instance

        Returns:
        -------
            set[str]: A copy of the set of rooms

        Processing Logic:
        ----------------
            - Creates a copy of the internal _rooms set to avoid direct manipulation of the original set
            - The copy() method is used to make a shallow copy of the set
            - This allows returning the rooms set without allowing external modification of the internal state
            - Returns the copy of the rooms set.
        """
        return copy(self._rooms)

    def add_room(
        self,
        model: str,
    ):
        """Adds a room. If an room already exists, it is ignored; no error is thrown.

        Args:
        ----
            model: The name or model of the room.
        """
        model = model.lower()

        if model not in self._rooms:
            self._visibility[model] = set()

        self._rooms.add(model)

    def remove_room(
        self,
        model: str,
    ):
        """Removes a room. If a room does not exist, it is ignored; no error is thrown.

        Args:
        ----
            model: The name or model of the room.
        """
        lower_model: str = model.lower()

        self._rooms.discard(lower_model)
        self._visibility.pop(lower_model, None)
        for observed in self._visibility.values():
            observed.discard(lower_model)

    def rename_room(
        self,
        old: str,
        new: str,
    ):
        """Rename a room and every reference to it, preserving declaration order.

        Raises ValueError without modifying the graph if the old room is
        missing or the new name belongs to another room.
        """
        old = old.lower()
        new = new.lower()
        if old == new:
            return
        if old not in self._rooms:
            raise ValueError(f"Room '{old}' does not exist.")
        if new in self._rooms:
            raise ValueError(f"Room '{new}' already exists.")

        self._visibility = {
            new if observer == old else observer: {
                new if room == old else room for room in observed
            }
            for observer, observed in self._visibility.items()
        }
        self._rooms.remove(old)
        self._rooms.add(new)

    def room_exists(
        self,
        model: str,
    ) -> bool:
        """Returns true if the specified room exists.

        Returns:
        -------
            True if the room exists.
        """
        return model.lower() in self._rooms

    def set_visible(
        self,
        when_inside: str,
        show: str,
        visible: bool,
    ):
        """Set or clear visibility in both directions between two rooms.

        Args:
        ----
            when_inside: The room of the observer.
            show: The observed room.
            visible: If the observed room is visible.
        """
        when_inside = when_inside.lower()
        show = show.lower()

        if when_inside not in self._rooms or show not in self._rooms:
            msg = "One of the specified rooms does not exist."
            raise ValueError(msg)

        if visible:
            self._visibility[when_inside].add(show)
            self._visibility[show].add(when_inside)
        else:
            self._visibility[when_inside].discard(show)
            self._visibility[show].discard(when_inside)

    def get_visible(
        self,
        when_inside: str,
        show: str,
    ) -> bool:
        """Returns true if the observed room is visible from the observing room.

        Args:
        ----
            when_inside: The room of the observer.
            show: The observed room.

        Returns:
        -------
            True if the room is visible.
        """
        when_inside = when_inside.lower()
        show = show.lower()

        if when_inside not in self._rooms or show not in self._rooms:
            msg = "One of the specified rooms does not exist."
            raise ValueError(msg)

        return show in self._visibility[when_inside]

    def set_all_visible(
        self,
    ):
        """Sets all rooms visible from each other.

        Processing Logic:
        ----------------
            - Loop through each room in self._rooms
            - For that room, loop through all other rooms
            - Set the visibility between the current room and other room to True.
        """
        for when_inside in self._rooms:
            for show in (room for room in self._rooms if room != when_inside):
                self.set_visible(when_inside, show, visible=True)
