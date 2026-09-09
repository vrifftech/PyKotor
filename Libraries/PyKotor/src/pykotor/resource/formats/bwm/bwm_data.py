from __future__ import annotations

import math
import struct

from collections import deque
from enum import IntEnum
from typing import TYPE_CHECKING

from pykotor.common.geometry import Face, Vector3

if TYPE_CHECKING:
    from collections.abc import Mapping

# A lot of the code in this module was adapted from the KotorBlender fork by seedhartha:
# https://github.com/seedhartha/kotorblender


class BWMType(IntEnum):
    PlaceableOrDoor = 0
    AreaModel = 1


class BWM:
    """A walkmesh with shared vertex identities and per-edge room transitions."""

    def __init__(
        self,
    ):
        self.walkmesh_type: BWMType = BWMType.AreaModel
        self.faces: list[BWMFace] = []
        # Retain the vertex table (including unused records) and the original
        # collision sections until geometry or walkability actually changes.
        self._vertices: list[Vector3] = []
        self._source_data: bytes | None = None
        self._source_geometry: tuple | None = None

        self.position: Vector3 = Vector3.from_null()
        self.relative_hook1: Vector3 = Vector3.from_null()
        self.relative_hook2: Vector3 = Vector3.from_null()
        self.absolute_hook1: Vector3 = Vector3.from_null()
        self.absolute_hook2: Vector3 = Vector3.from_null()

    def walkable_faces(
        self,
    ) -> list[BWMFace]:
        """Get a list of walkable faces'.

        Args:
        ----
            self: Object containing faces

        Returns:
        -------
            list[BWMFace]: List of faces that are walkable

        Processing Logic:
        ----------------
            - Iterate through all faces in self.faces
            - Check if each face's material is walkable using face.material.walkable()
            - Add face to return list if walkable.
        """
        return [face for face in self.faces if face.material.walkable()]

    def unwalkable_faces(
        self,
    ) -> list[BWMFace]:
        """Return unwalkable faces in the mesh.

        Args:
        ----
            self: The mesh object

        Returns:
        -------
            list[BWMFace]: List of unwalkable faces in the mesh

        Processing Logic:
        ----------------
            - Iterate through all faces in the mesh
            - Check if the material of the face is not walkable
            - Add the face to the return list if material is not walkable
            - Return the list of unwalkable faces.
        """
        return [face for face in self.faces if not face.material.walkable()]

    def vertices(self) -> list[Vector3]:
        """Return vertex records by identity, never welding equal coordinates."""
        vertices: list[Vector3] = []
        seen: set[int] = set()
        for vertex in self._vertices:
            if id(vertex) not in seen:
                vertices.append(vertex)
                seen.add(id(vertex))
        for face in self.faces:
            for vertex in (face.v1, face.v2, face.v3):
                if id(vertex) not in seen:
                    vertices.append(vertex)
                    seen.add(id(vertex))
        return vertices

    def _geometry_signature(self) -> tuple:
        """Snapshot the geometry, topology and walkability used by collision data."""
        vertices = self.vertices()
        indices = {id(vertex): i for i, vertex in enumerate(vertices)}
        return (
            self.walkmesh_type,
            b"".join(struct.pack("<3f", v.x, v.y, v.z) for v in vertices),
            tuple(tuple(indices[id(v)] for v in (f.v1, f.v2, f.v3)) for f in self.faces),
            tuple(f.material.walkable() for f in self.faces),
        )

    def aabbs(self) -> list[BWMNodeAABB]:
        """Build a balanced collision tree without depth or coordinate cutoffs.

        Stable median partitions also handle distinct faces with equal centroids.
        Child references point to actual nodes in the returned preorder table.
        """
        nodes: list[BWMNodeAABB] = []
        pending: list[tuple[list[BWMFace], BWMNodeAABB | None, bool]] = []
        if self.faces:
            pending.append((list(self.faces), None, False))
        while pending:
            faces, parent, right = pending.pop()
            points = [v for face in faces for v in (face.v1, face.v2, face.v3)]
            minimum = Vector3(*(min(v[axis] for v in points) for axis in range(3)))
            maximum = Vector3(*(max(v[axis] for v in points) for axis in range(3)))
            axis = max(range(3), key=lambda i: maximum[i] - minimum[i])
            leaf = len(faces) == 1
            node = BWMNodeAABB(
                minimum, maximum, faces[0] if leaf else None,
                0 if leaf else axis + 1, None, None,
            )
            nodes.append(node)
            if parent is not None:
                if right:
                    parent.right = node
                else:
                    parent.left = node
            if not leaf:
                ordered = sorted(faces, key=lambda face: face.centre()[axis])
                middle = len(ordered) // 2
                pending.append((ordered[middle:], node, True))
                pending.append((ordered[:middle], node, False))
        return nodes

    def _adjacency_map(self) -> dict[BWMFace, tuple[BWMAdjacency | None, ...]]:
        """Connect walkable faces through shared vertex records, not coordinates."""
        rows: dict[BWMFace, list[BWMAdjacency | None]] = {
            face: [None, None, None] for face in self.walkable_faces()
        }
        edges: dict[tuple[int, int], list[tuple[BWMFace, int]]] = {}
        for face in rows:
            vertices = (face.v1, face.v2, face.v3)
            for edge in range(3):
                a, b = id(vertices[edge]), id(vertices[(edge + 1) % 3])
                key = (min(a, b), max(a, b))
                edges.setdefault(key, []).append((face, edge))
        for pairs in edges.values():
            if len(pairs) > 2:
                raise ValueError("A walkmesh edge has more than two walkable incident faces.")
            if len(pairs) == 2:
                (first, a), (second, b) = pairs
                if first is second:
                    raise ValueError("A walkmesh face repeats the same vertex edge.")
                rows[first][a] = BWMAdjacency(second, b)
                rows[second][b] = BWMAdjacency(first, a)
        return {face: tuple(row) for face, row in rows.items()}

    def edges(self) -> list[BWMEdge]:
        """Order boundary edges into closed perimeters using vertex-record IDs.

        At a shared boundary vertex, use the first remaining edge in face/edge
        order, as the engine's room-border builder does. Indices stay local.
        """
        adjacency = self._adjacency_map()
        boundary: list[tuple[BWMFace, int]] = [
            (face, edge) for face, row in adjacency.items()
            for edge, neighbor in enumerate(row) if neighbor is None
        ]
        outgoing: dict[int, deque[int]] = {}
        for index, (face, edge) in enumerate(boundary):
            vertex = (face.v1, face.v2, face.v3)[edge]
            outgoing.setdefault(id(vertex), deque()).append(index)
        visited: set[int] = set()
        edges: list[BWMEdge] = []
        for origin in range(len(boundary)):
            if origin in visited:
                continue
            first, local = boundary[origin]
            start = (first.v1, first.v2, first.v3)[local]
            current = origin
            while True:
                face, edge = boundary[current]
                visited.add(current)
                transition = (face.trans1, face.trans2, face.trans3)[edge]
                edges.append(BWMEdge(face, edge, -1 if transition is None else transition))
                end = (face.v1, face.v2, face.v3)[(edge + 1) % 3]
                if end is start:
                    edges[-1].final = True
                    break
                candidates = outgoing.get(id(end))
                if candidates is not None:
                    while candidates and candidates[0] in visited:
                        candidates.popleft()
                if not candidates:
                    raise ValueError("Walkmesh boundary edges do not form a closed perimeter.")
                current = candidates.popleft()
        return edges

    def adjacencies(
        self,
        face: BWMFace,
    ) -> tuple[BWMAdjacency | None, BWMAdjacency | None, BWMAdjacency | None]:
        """Return the neighboring walkable face and its local edge for each edge."""
        row = self._adjacency_map().get(face, (None, None, None))
        return row[0], row[1], row[2]

    def box(
        self,
    ) -> tuple[Vector3, Vector3]:
        """Calculates bounding box of the mesh.

        Args:
        ----
            self: Mesh object

        Returns:
        -------
            tuple[Vector3, Vector3]: Bounding box minimum and maximum points

        Processing Logic:
        ----------------
            - Initialize bounding box minimum and maximum points to extreme values
            - Iterate through all vertices of the mesh
            - Update minimum x, y, z values of bbmin
            - Update maximum x, y, z values of bbmax
            - Return bounding box minimum and maximum points.
        """
        vertices = self.vertices()
        if not vertices:
            return Vector3.from_null(), Vector3.from_null()
        bbmin = Vector3(math.inf, math.inf, math.inf)
        bbmax = Vector3(-math.inf, -math.inf, -math.inf)
        for vertex in vertices:
            self._handle_vertex(bbmin, vertex, bbmax)
        return bbmin, bbmax

    def _handle_vertex(self, bbmin: Vector3, vertex: Vector3, bbmax: Vector3):
        """Update bounding box with vertex position.

        Args:
        ----
            bbmin: Vector3 - Bounding box minimum point
            vertex: Vector3 - Vertex position
            bbmax: Vector3 - Bounding box maximum point

        Returns:
        -------
            None - Updates bbmin and bbmax in place

        Processing Logic:
        ----------------
            - Compare vertex x, y, z to bbmin x, y, z and update bbmin with minimum
            - Compare vertex x, y, z to bbmax x, y, z and update bbmax with maximum.
        """
        bbmin.x = min(bbmin.x, vertex.x)
        bbmin.y = min(bbmin.y, vertex.y)
        bbmin.z = min(bbmin.z, vertex.z)
        bbmax.x = max(bbmax.x, vertex.x)
        bbmax.y = max(bbmax.y, vertex.y)
        bbmax.z = max(bbmax.z, vertex.z)

    def faceAt(
        self,
        x: float,
        y: float,
    ) -> BWMFace | None:
        """Returns the face at the given 2D coordinates if there is one otherwise returns None.

        Args:
        ----
            x: The x coordinate.
            y: The y coordinate.

        Returns:
        -------
            BWMFace object or None.
        """
        for face in self.faces:
            v1 = face.v1
            v2 = face.v2
            v3 = face.v3

            # Degenerate XY projections have no unique height. On shared
            # edges/vertices, the first containing face in source order wins.
            area = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)
            if area == 0:
                continue
            c1 = (v2.x - v1.x) * (y - v1.y) - (v2.y - v1.y) * (x - v1.x)
            c2 = (v3.x - v2.x) * (y - v2.y) - (v3.y - v2.y) * (x - v2.x)
            c3 = (v1.x - v3.x) * (y - v3.y) - (v1.y - v3.y) * (x - v3.x)

            if (c1 <= 0 and c2 <= 0 and c3 <= 0) or (c1 >= 0 and c2 >= 0 and c3 >= 0):
                return face
        return None

    def translate(
        self,
        x: float,
        y: float,
        z: float,
    ):
        """Shifts the position of the walkmesh.

        Args:
        ----
            x: How many units to shift on the X-axis.
            y: How many units to shift on the Y-axis.
            z: How many units to shift on the Z-axis.
        """
        for vertex in self.vertices():
            vertex.x += x
            vertex.y += y
            vertex.z += z

    def rotate(
        self,
        degrees: float,
    ):
        """Rotates the walkmesh around the Z-axis counter-clockwise.

        Args:
        ----
            degrees: The angle to rotate in degrees.
        """
        radians = math.radians(degrees)
        cos = math.cos(radians)
        sin = math.sin(radians)

        for vertex in self.vertices():
            x, y = vertex.x, vertex.y
            vertex.x = x * cos - y * sin
            vertex.y = x * sin + y * cos

    def change_lyt_indexes(self, old: int, new: int | None):
        """Replace a single room index on all three edges."""
        self.remap_transitions({old: new})

    def remap_transitions(self, mapping: Mapping[int, int | None]):
        """Substitute room indices once, using each edge's original value."""
        for face in self.faces:
            face.trans1, face.trans2, face.trans3 = (
                mapping.get(value, value) if value is not None else None
                for value in (face.trans1, face.trans2, face.trans3)
            )

    def flip(
        self,
        x: bool,  # noqa: FBT001
        y: bool,  # noqa: FBT001
    ):
        """Flips the walkmesh around the specified axes.

        Args:
        ----
            x: Flip around the X-axis.
            y: Flip around the Y-axis.
        """
        if not x and not y:
            return

        for vertex in self.vertices():
            if x:
                vertex.x = -vertex.x
            if y:
                vertex.y = -vertex.y

        # Fix the face normals
        if bool(x) != bool(y):
            for face in self.faces:
                v1, v2, v3 = face.v1, face.v2, face.v3
                face.v1, face.v2, face.v3 = v3, v2, v1
                # Reversed winding exchanges the first two geometric edges.
                face.trans1, face.trans2 = face.trans2, face.trans1


class BWMFace(Face):
    """An extension of the Face class with a transition index for each edge."""

    def __init__(
        self,
        v1: Vector3,
        v2: Vector3,
        v3: Vector3,
    ):
        super().__init__(v1, v2, v3)
        self.trans1: int | None = None
        self.trans2: int | None = None
        self.trans3: int | None = None


class BWMMostSignificantPlane(IntEnum):
    NEGATIVE_Z = -3
    NEGATIVE_Y = -2
    NEGATIVE_X = -1
    NONE = 0
    POSITIVE_X = 1
    POSITIVE_Y = 2
    POSITIVE_Z = 3


class BWMNodeAABB:
    """A node in an AABB tree. Calculated with BWM.aabbs()."""

    def __init__(
        self,
        bb_min: Vector3,
        bb_max: Vector3,
        face: BWMFace | None,
        sigplane: int,
        left: BWMNodeAABB | None,
        right: BWMNodeAABB | None,
    ):
        """Initializes a bounding volume node.

        Args:
        ----
            bb_min: Vector3 - Minimum bounds of the bounding box
            bb_max: Vector3 - Maximum bounds of the bounding box
            face: BWMFace | None - Face that splits the node or None
            sigplane: int - Index of most significant splitting plane
            left: BWMNodeAABB | None - Left child node or None
            right: BWMNodeAABB | None - Right child node or None

        Returns:
        -------
            self - The initialized BWMNodeAABB object

        Processing Logic:
        ----------------
            - Sets the bounding box minimum and maximum bounds
            - Sets the splitting face and most significant plane
            - Sets the left and right child nodes.
        """
        self.bb_min: Vector3 = bb_min
        self.bb_max: Vector3 = bb_max
        self.face: BWMFace | None = face
        self.sigplane: BWMMostSignificantPlane = BWMMostSignificantPlane(sigplane)
        self.left: BWMNodeAABB | None = left
        self.right: BWMNodeAABB | None = right


class BWMAdjacency:
    """Maps a edge index (0 to 2 inclusive) to a target face from a source face. Calculated with BWM.adjacencies().

    Attributes:
    ----------
        face: Target face.
        edge: Edge index of the source face (0 to 2 inclusive).
    """

    def __init__(
        self,
        face: BWMFace,
        index: int,
    ):
        self.face: BWMFace = face
        self.edge: int = index


class BWMEdge:
    """Represents an edge of a the face that is not adjacent to any other walkable face. Calculated with BWM.edges().

    Attributes:
    ----------
        face: The face.
        index: Edge index on the face (0 to 2 inclusive).
        transition: Index into a LYT file.
        final: This is the final edge of the perimeter.
    """

    def __init__(
        self,
        face: BWMFace,
        index: int,
        transition: int,
        *,
        final: bool = False,
    ):
        self.face: BWMFace = face
        self.index: int = index
        self.transition: int = transition
        self.final: bool = final
