from __future__ import annotations

import struct

from typing import TYPE_CHECKING

from pykotor.common.geometry import SurfaceMaterial, Vector3
from pykotor.resource.formats.bwm.bwm_data import BWM, BWMFace, BWMType
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


class BWMBinaryReader(ResourceReader):
    def __init__(self, source: SOURCE_TYPES, offset: int = 0, size: int = 0):
        super().__init__(source, offset, size)
        self._wok: BWM | None = None

    @autoclose
    def load(self, auto_close: bool = True) -> BWM:
        data = self._reader.read_bytes(self._size)
        if len(data) < 136 or data[:4] != b"BWM ":
            raise ValueError("Not a valid binary BWM file.")
        if data[4:8] != b"V1.0":
            raise ValueError("The BWM version of the file is unsupported.")

        wok = self._wok = BWM()
        wok.walkmesh_type = BWMType(struct.unpack_from("<I", data, 8)[0])
        vectors = [Vector3(*struct.unpack_from("<3f", data, offset)) for offset in range(12, 72, 12)]
        wok.relative_hook1, wok.relative_hook2, wok.absolute_hook1, wok.absolute_hook2, wok.position = vectors
        (vertex_count, vertex_offset, face_count, face_offset, material_offset,
         normal_offset, plane_offset, aabb_count, aabb_offset, root,
         adjacency_count, adjacency_offset, edge_count, edge_offset,
         perimeter_count, perimeter_offset) = struct.unpack_from("<16I", data, 72)

        # Validate table extents before dereferencing any file-provided offset.
        for count, offset, stride in (
            (vertex_count, vertex_offset, 12), (face_count, face_offset, 12),
            (face_count, material_offset, 4), (face_count, normal_offset, 12),
            (face_count, plane_offset, 4), (aabb_count, aabb_offset, 44),
            (adjacency_count, adjacency_offset, 12), (edge_count, edge_offset, 8),
            (perimeter_count, perimeter_offset, 4),
        ):
            if count and (offset < 136 or offset + count * stride > len(data)):
                raise ValueError("A BWM table extends outside the resource.")
        if aabb_count and root >= aabb_count:
            raise ValueError("The BWM tree root is outside the node table.")
        if adjacency_count > face_count:
            raise ValueError("The BWM adjacency table has more rows than faces.")

        vertices = [Vector3(*struct.unpack_from("<3f", data, vertex_offset + i * 12)) for i in range(vertex_count)]
        for i in range(face_count):
            indices = struct.unpack_from("<3I", data, face_offset + i * 12)
            if any(index >= vertex_count for index in indices):
                raise ValueError("A BWM face references a missing vertex.")
            face = BWMFace(*(vertices[index] for index in indices))
            face.material = SurfaceMaterial(struct.unpack_from("<I", data, material_offset + i * 4)[0])
            wok.faces.append(face)
        for i in range(edge_count):
            edge, transition = struct.unpack_from("<2I", data, edge_offset + i * 8)
            if edge >= face_count * 3:
                raise ValueError("A BWM boundary references a missing face edge.")
            face, local = divmod(edge, 3)
            if transition != 0xFFFFFFFF:
                setattr(wok.faces[face], ("trans1", "trans2", "trans3")[local], transition)
        previous = 0
        for i in range(perimeter_count):
            end = struct.unpack_from("<I", data, perimeter_offset + i * 4)[0]
            if not previous < end <= edge_count:
                raise ValueError("A BWM perimeter has an invalid end index.")
            previous = end

        wok._vertices = vertices
        wok._source_data = data
        wok._source_geometry = wok._geometry_signature()
        return wok


class BWMBinaryWriter(ResourceWriter):
    HEADER_SIZE = 136

    def __init__(self, wok: BWM, target: TARGET_TYPES):
        self._wok = wok
        # Finish geometry generation and encoding before opening even a direct
        # writer's destination. Public path writes additionally commit atomically.
        self._data = self._build()
        super().__init__(target)

    @autoclose
    def write(self, auto_close: bool = True):
        self._writer.write_bytes(self._data)

    def _build(self) -> bytes:
        wok = self._wok
        if wok._source_data is not None and wok._geometry_signature() == wok._source_geometry:
            # Keep original collision structures, normals, padding and table order
            # when only hooks, position, material properties or transitions changed.
            data = bytearray(wok._source_data)
            self._write_properties(data)
            material_offset = struct.unpack_from("<I", data, 88)[0]
            for index, face in enumerate(wok.faces):
                struct.pack_into("<I", data, material_offset + index * 4, face.material.value)
            edge_count, edge_offset = struct.unpack_from("<2I", data, 120)
            for index in range(edge_count):
                edge = struct.unpack_from("<I", data, edge_offset + index * 8)[0]
                face, local = divmod(edge, 3)
                value = (wok.faces[face].trans1, wok.faces[face].trans2, wok.faces[face].trans3)[local]
                struct.pack_into("<I", data, edge_offset + index * 8 + 4,
                                 0xFFFFFFFF if value is None or value == -1 else value)
            return bytes(data)

        vertices = wok.vertices()
        vertex_indices = {id(vertex): index for index, vertex in enumerate(vertices)}
        walkable = wok.walkable_faces()
        faces = walkable + wok.unwalkable_faces()
        face_indices = {face: index for index, face in enumerate(faces)}
        nodes = wok.aabbs()
        node_indices = {node: index for index, node in enumerate(nodes)}
        adjacency = wok._adjacency_map()
        edges = wok.edges()
        data = bytearray(self.HEADER_SIZE)
        data[:8] = b"BWM V1.0"
        self._write_properties(data)

        def block(rows, fmt: str) -> int:
            offset = len(data)
            for row in rows:
                data.extend(struct.pack("<" + fmt, *row))
            return offset

        vertex_offset = block(((v.x, v.y, v.z) for v in vertices), "3f")
        face_offset = block(((vertex_indices[id(f.v1)], vertex_indices[id(f.v2)], vertex_indices[id(f.v3)]) for f in faces), "3I")
        material_offset = block(((f.material.value,) for f in faces), "I")
        normals = [face.normal() for face in faces]
        normal_offset = block(((n.x, n.y, n.z) for n in normals), "3f")
        plane_offset = block(((-n.dot(f.v1),) for f, n in zip(faces, normals)), "f")
        aabb_offset = block((
            (n.bb_min.x, n.bb_min.y, n.bb_min.z, n.bb_max.x, n.bb_max.y, n.bb_max.z,
             0xFFFFFFFF if n.face is None else face_indices[n.face], 4, n.sigplane.value,
             0xFFFFFFFF if n.left is None else node_indices[n.left],
             0xFFFFFFFF if n.right is None else node_indices[n.right]) for n in nodes
        ), "6f5I")
        adjacency_offset = block((
            tuple(0xFFFFFFFF if a is None else face_indices[a.face] * 3 + a.edge for a in adjacency[face])
            for face in walkable
        ), "3I")
        edge_offset = block((
            (face_indices[e.face] * 3 + e.index, 0xFFFFFFFF if e.transition == -1 else e.transition) for e in edges
        ), "2I")
        perimeters = [index + 1 for index, edge in enumerate(edges) if edge.final]
        perimeter_offset = block(((end,) for end in perimeters), "I")
        struct.pack_into("<16I", data, 72, len(vertices), vertex_offset, len(faces), face_offset,
                         material_offset, normal_offset, plane_offset, len(nodes), aabb_offset, 0,
                         len(walkable), adjacency_offset, len(edges), edge_offset, len(perimeters), perimeter_offset)
        return bytes(data)

    def _write_properties(self, data: bytearray):
        wok = self._wok
        struct.pack_into("<I", data, 8, wok.walkmesh_type.value)
        vectors = (wok.relative_hook1, wok.relative_hook2, wok.absolute_hook1, wok.absolute_hook2, wok.position)
        for index, vector in enumerate(vectors):
            for axis, value in enumerate((vector.x, vector.y, vector.z)):
                offset = 12 + index * 12 + axis * 4
                encoded = struct.pack("<f", value)
                if wok._source_data is not None:
                    original = wok._source_data[offset:offset + 4]
                    # Preserve untouched float bit patterns, including NaN payloads.
                    if encoded == struct.pack("<f", struct.unpack("<f", original)[0]):
                        encoded = original
                data[offset:offset + 4] = encoded
