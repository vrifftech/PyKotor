from __future__ import annotations
import math
import struct
from typing import NamedTuple
from pykotor.common.geometry import Vector3, Vector4
from pykotor.common.misc import Game
from pykotor.resource.formats.mdl.io_mdl import MDLBinaryReader, MDLBinaryWriter, _Buffer, _GEOMETRY_POINTERS
from pykotor.resource.formats.mdl.mdl_data import multiply_quaternions, rotate_vector

class MDLMDXTuple(NamedTuple):
    mdl: bytes
    mdx: bytes

def _body(data):
    if len(data) < 208 or data[:4] != bytes(4):
        raise ValueError('Not a complete binary model')
    size = struct.unpack_from('<I', data, 4)[0]
    if size + 12 > len(data):
        raise ValueError('Model body exceeds its resource')
    return (_Buffer(data[12:12 + size]), data[12 + size:])

def _nodes(b):
    roots = [b.u32(40)]
    roots.extend((b.u32(a[0] + 40) for a in b.array(b.u32(88), b.u32(92), 'I')))
    for root in roots:
        pending, seen = ([root], set())
        while pending:
            p = pending.pop()
            if p in seen:
                raise ValueError('Cycle or shared-parent node in model')
            seen.add(p)
            b.read(p, 80)
            yield p
            pending.extend((a[0] for a in reversed(b.array(b.u32(p + 44), b.u32(p + 48), 'I'))))

def rename(data: bytes, name: str) -> bytes:
    b, tail = _body(data)
    b.text_at(8, 32, name)
    return data[:12] + bytes(b.data) + tail

def _references(data, offsets):
    b, _ = _body(data)
    seen = set()
    for p in _nodes(b):
        if not b.values(p, 'H')[0] & 32:
            continue
        for off in offsets:
            name = b.text(p + 80 + off, 32).lower()
            if name and name != 'null' and (name not in seen):
                seen.add(name)
                yield name

def iterate_textures_and_lightmaps(data: bytes):
    yield from _references(data, (88, 120))

def iterate_textures(data: bytes):
    yield from _references(data, (88,))

def iterate_lightmaps(data: bytes):
    yield from _references(data, (120,))

def _rename_references(data, mapping, offset):
    b, tail = _body(data)
    names = {old.lower(): new.lower() for old, new in mapping.items()}
    # Validate all requests before changing any stored field.
    for name in names.values():
        raw = name.encode('latin1')
        if len(raw) > 32 or b'\x00' in raw:
            raise ValueError('Texture names must fit a 32-byte field')
    for p in _nodes(b):
        if b.values(p, 'H')[0] & 32:
            at = p + 80 + offset
            name = b.text(at, 32).lower()
            if name in names:
                b.text_at(at, 32, names[name])
    return data[:12] + bytes(b.data) + tail

def change_textures(data: bytes, textures: dict[str, str]) -> bytes:
    return _rename_references(data, textures, 88)

def change_lightmaps(data: bytes, textures: dict[str, str]) -> bytes:
    return _rename_references(data, textures, 120)

def detect_version(data: bytes) -> Game:
    b, _ = _body(data)
    for game, pointers in _GEOMETRY_POINTERS.items():
        if b.u32(0) == pointers[0]:
            return game
    raise ValueError('Unknown model layout')

def _convert(data, game):
    if detect_version(data) == game:
        return data
    model = MDLBinaryReader(data).load()
    writer = MDLBinaryWriter(model, None)
    writer.game = game
    return writer.encode()[0]

def convert_to_k1(data: bytes) -> bytes:
    """Relocate every base/animation node and subtype without changing MDX addresses."""
    return _convert(data, Game.K1)

def convert_to_k2(data: bytes) -> bytes:
    return _convert(data, Game.K2)

def _bounds(lo, hi, operation):
    points = [operation(Vector3(x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    return (Vector3(*(min((getattr(v, a) for v in points)) for a in ('x', 'y', 'z'))), Vector3(*(max((getattr(v, a) for v in points)) for a in ('x', 'y', 'z'))))

def transform(data: bytes, translation: Vector3, rotation: float) -> bytes:
    """Apply a rigid world transform to the base and animation roots.

    Vertex streams and internal node/bone identities do not change. Controllers
    that override a root's transform are composed with the same operation.
    """
    if rotation == 0 and all((v == 0 for v in translation)):
        return data
    model = MDLBinaryReader(data).load()
    angle = math.radians(rotation) * 0.5
    q = Vector4(0, 0, math.sin(angle), math.cos(angle))

    def move(v):
        r = rotate_vector(q, v)
        return Vector3(r.x + translation.x, r.y + translation.y, r.z + translation.z)
    for graph in [model] + model.anims:
        n = graph.root
        n.position = move(n.position)
        n.orientation = multiply_quaternions(q, n.orientation)
        for c in n.controllers:
            kind = c.controller_type & 16777215
            for row in c.rows:
                if kind == 8:
                    for i in range(0, len(row.data), 3):
                        value = Vector3(*row.data[i:i + 3])
                        # Bézier handles are offsets from the key, not positions.
                        row.data[i:i + 3] = list(move(value) if i == 0 else rotate_vector(q, value))
                elif kind == 20:
                    row.data = list(multiply_quaternions(q, Vector4(*row.data)))
    model.bb_min, model.bb_max = _bounds(model.bb_min, model.bb_max, move)
    return MDLBinaryWriter(model, None).encode()[0]

def flip(mdl_data: bytes, mdx_data: bytes, *, flip_x: bool, flip_y: bool) -> MDLMDXTuple:
    """Reflect complete node/bone transforms and geometry in model space.

    A reflection S conjugates each local rotation (S R S); reflecting only local
    vertices is not a world-space mirror when nodes are rotated.
    """
    if not flip_x and (not flip_y):
        return MDLMDXTuple(mdl_data, mdx_data)
    model = MDLBinaryReader(mdl_data, source_ext=mdx_data).load()
    sx, sy = (-1 if flip_x else 1, -1 if flip_y else 1)
    det = sx * sy

    def reflect(v):
        return Vector3(sx * v.x, sy * v.y, v.z)

    def orient(q):
        return Vector4(det * sx * q.x, det * sy * q.y, det * q.z, q.w)
    model.bb_min, model.bb_max = _bounds(model.bb_min, model.bb_max, reflect)
    for graph in [model] + model.anims:
        for n in graph.all_nodes():
            n.position = reflect(n.position)
            n.orientation = orient(n.orientation)
            for c in n.controllers:
                for row in c.rows:
                    if c.controller_type & 16777215 == 8:
                        for i in range(0, len(row.data), 3):
                            row.data[i:i + 3] = list(reflect(Vector3(*row.data[i:i + 3])))
                    elif c.controller_type & 16777215 == 20:
                        row.data = list(orient(Vector4(*row.data)))
            if n.mesh:
                m = n.mesh
                m.vertex_positions = [reflect(v) for v in m.vertex_positions]
                if m.vertex_normals is not None:
                    m.vertex_normals = [reflect(v) for v in m.vertex_normals]
                m.bb_min, m.bb_max = _bounds(m.bb_min, m.bb_max, reflect)
                m.average = reflect(m.average)
                if m.mdx_bitmap & 128:
                    raw = bytearray(m.mdx_data)
                    start = m.mdx_offsets[7]
                    if start == 4294967295 or start + 36 > m.mdx_stride:
                        raise ValueError('Invalid tangent-frame MDX range')
                    for i in range(len(m.vertex_positions)):
                        for axis in range(3):
                            at = i * m.mdx_stride + start + 12 * axis
                            struct.pack_into('<3f', raw, at, *reflect(Vector3(*struct.unpack_from('<3f', raw, at))))
                    m.mdx_data = bytes(raw)
                for f in m.faces:
                    f.normal = reflect(f.normal)
                    if det < 0:
                        f.v2, f.v3 = (f.v3, f.v2)
                        f.a1, f.a3 = (f.a3, f.a1)
                if det < 0:
                    for g in m.indices:
                        if len(g) % 3:
                            raise ValueError('Triangle index group is not a multiple of three')
                        for i in range(0, len(g), 3):
                            g[i + 1], g[i + 2] = (g[i + 2], g[i + 1])
            if n.skin:
                n.skin.qbones = [orient(q) for q in n.skin.qbones]
                n.skin.tbones = [reflect(v) for v in n.skin.tbones]
            if n.dangly:
                n.dangly.vertices = [reflect(v) for v in n.dangly.vertices]
            if n.saber:
                n.saber.vertices = [reflect(v) for v in n.saber.vertices]
                n.saber.normals = [reflect(v) for v in n.saber.normals]
            if n.aabb and n.aabb.root:
                pending = [n.aabb.root]
                while pending:
                    a = pending.pop()
                    a.bb_min, a.bb_max = _bounds(a.bb_min, a.bb_max, reflect)
                    pending.extend((c for c in (a.left, a.right) if c is not None))
    d, x = MDLBinaryWriter(model, None).encode()
    return MDLMDXTuple(d, x)
