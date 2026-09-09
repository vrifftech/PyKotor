"""This module handles classes relating to editing RIM files."""

from __future__ import annotations

from copy import copy
from typing import Any, Generator

from pykotor.common.misc import ResRef
from pykotor.extract.file import ResourceIdentifier
from pykotor.resource.formats.erf.erf_data import ERF
from pykotor.resource.type import ResourceType


class RIM:
    """Represents the data of a RIM file."""

    BINARY_TYPE = ResourceType.RIM

    def __init__(
        self,
    ):
        self._resources: list[RIMResource] = []
        self.unknown: int = 0
        self.reserved: bytes = bytes(100)

    def __iter__(
        self,
    ) -> Generator[RIMResource, Any, None]:
        """Iterates through the stored resources yielding a copied resource each iteration."""
        for resource in self._resources:
            yield copy(resource)

    def __len__(
        self,
    ):
        """Returns the number of stored resources."""
        return len(self._resources)

    def __getitem__(
        self,
        item,
    ):
        """Returns a resource at the specified index or with the specified resref."""
        if isinstance(item, int):
            return self._resources[item]
        if isinstance(item, str):
            try:
                return next(resource for resource in self._resources if resource.resref == item)
            except StopIteration as e:
                msg = f"{item} not found."
                raise KeyError(msg) from e
        return NotImplemented

    def __add__(self, other: RIM) -> RIM:
        """Combines the resources of two RIM instances into a new RIM instance.

        Args:
        ----
            other: Another RIM instance.

        Returns:
        -------
            A new RIM instance containing the combined resources.
        """
        if not isinstance(other, RIM):
            return NotImplemented

        combined_rim = RIM()
        replaced = {resource.identifier() for resource in other}
        for resource in self:
            if resource.identifier() not in replaced:
                combined_rim.append(RIMResource(ResRef.from_bytes(resource.resref.to_bytes()), resource.restype, resource.data))
        for resource in other:
            combined_rim.append(RIMResource(ResRef.from_bytes(resource.resref.to_bytes()), resource.restype, resource.data))

        return combined_rim

    def append(self, resource: RIMResource) -> None:
        """Appends a physical record without replacing an earlier matching key."""
        if not 0 <= resource.restype.type_id <= 0xFFFFFFFF:
            raise ValueError(f"Invalid RIM resource type ID: {resource.restype.type_id}")
        self._resources.append(resource)

    def set_data(
        self,
        resname: str,
        restype: ResourceType,
        data: bytes,
    ):
        """Sets the data of the resource with the specified resref/restype pair.

        If it does not exists, a resource is appended to the resource list.

        Args:
        ----
            resname: The resource reference filename.
            restype: The resource type.
            data: The new resource data.
        """
        if not 0 <= restype.type_id <= 0xFFFFFFFF:
            raise ValueError(f"Invalid RIM resource type ID: {restype.type_id}")

        resource: RIMResource | None = next(
            (resource for resource in self._resources if resource.resref == resname and resource.restype == restype),
            None,
        )
        if resource is None:
            self._resources.append(RIMResource(ResRef(resname), restype, data))
        else:
            resource.data = bytes(data)

    def get(
        self,
        resname: str,
        restype: ResourceType,
    ) -> bytes | None:
        """Returns the data of the resource with the specified resref/restype pair if it exists, otherwise returns None.

        Args:
        ----
            resname: The resource reference filename.
            restype: The resource type.

        Returns:
        -------
            The bytes data of the resource or None.
        """
        resource: RIMResource | None = next(
            (resource for resource in self._resources if resource.resref == resname and resource.restype == restype),
            None,
        )
        return None if resource is None else resource.data

    def remove(
        self,
        resname: str,
        restype: ResourceType,
    ):
        """Removes all records for the given key so no hidden duplicate becomes active.

        Args:
        ----
            resname: The resource reference filename.
            restype: The resource type.
        """
        key = ResourceIdentifier(resname, restype)
        self._resources[:] = [resource for resource in self._resources if resource.identifier() != key]

    def to_erf(
        self,
    ):
        """Returns a ERF with the same resources. Defaults to an ERF with ERFType.ERF set.

        Returns:
        -------
            A new ERF object.
        """
        from pykotor.resource.formats.erf import ERF, ERFResource  # Prevent circular imports

        erf = ERF()
        for resource in self._resources:
            erf.append(ERFResource(ResRef.from_bytes(resource.resref.to_bytes()), resource.restype, resource.data))
        return erf

    def __eq__(self, other):
        from pykotor.resource.formats.rim import RIM
        if not isinstance(other, (ERF, RIM)):
            return NotImplemented
        return self._resources == other._resources


class RIMResource:
    def __init__(
        self,
        resref: ResRef,
        restype: ResourceType,
        data: bytes,
    ):
        self.resref: ResRef = resref
        self.restype: ResourceType = restype
        if isinstance(data, bytearray):  # FIXME: Something is passing bytearray here
            data = bytes(data)
        self.data: bytes = data

    def __eq__(
        self,
        other,
    ):
        from pykotor.resource.formats.erf import ERFResource
        if not isinstance(other, (ERFResource, RIMResource)):
            return NotImplemented
        return (
            self.resref == other.resref
            and self.restype == other.restype
            and self.data == other.data
        )

    def __hash__(self):
        return hash((self.resref, self.restype, self.data))

    def identifier(self) -> ResourceIdentifier:
        return ResourceIdentifier(str(self.resref), self.restype)
