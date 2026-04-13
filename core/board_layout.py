from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, TypeAlias

from core.constants import (
    NUMBER_TOKEN_COUNTS,
    NUMBER_TOKEN_ORDER_FIXED,
    NUM_CONNECTIONS,
    NUM_TILES,
    NUM_VERTICES,
    PORT_COUNTS,
    PlayerId,
    PortType,
    Resource,
    TILE_RESOURCE_COUNTS,
)
from core.connection import Connection
from core.hex_tile import HexTile
from core.port import Port
from core.vertex import Vertex

Coord: TypeAlias = Tuple[int, int]
VertexKey: TypeAlias = Tuple[Coord, ...] | Tuple[str, Coord, int]


class BoardLayout:
    """
    Board topology + tile/vertex/edge/port structure for standard 4-player Catan.

    This supports:
    - fixed setup token ordering
    - variable setup token ordering
    - desert + robber
    - 54 vertices / 72 edges
    """

    def __init__(self, seed: Optional[int] = None, fixed_setup: bool = False):
        if seed is not None:
            random.seed(seed)

        self.fixed_setup = fixed_setup

        self.tiles: List[HexTile] = []
        self.vertices: List[Vertex] = []
        self.connections: List[Connection] = []
        self.ports: List[Port] = []

        self.tile_coords: List[Coord] = []
        self.vertex_to_ports: Dict[int, Port] = {}

        self._build_board()

    def _build_board(self) -> None:
        self._create_tiles()
        self._create_vertices()
        self._assign_tile_vertices()
        self._create_connections()
        self._assign_vertex_neighbors()
        self._create_ports()

    def _create_tiles(self) -> None:
        resources: List[Resource] = []
        for resource, count in TILE_RESOURCE_COUNTS.items():
            resources.extend([resource] * count)

        self.tile_coords = [
            (q, r)
            for q in range(-2, 3)
            for r in range(-2, 3)
            if -2 <= q + r <= 2
        ]

        if len(resources) != NUM_TILES:
            raise ValueError(f"Expected {NUM_TILES} resources, got {len(resources)}")

        if self.fixed_setup:
            # Fixed setup: keep the desert among the 19 hexes, but use the official
            # fixed number ordering for the 18 non-desert tiles.
            # The exact fixed terrain arrangement can be swapped in later if you want
            # page-4 matching geometry; this version already fixes the token ordering.
            random.shuffle(resources)
        else:
            random.shuffle(resources)

        numbers: List[int] = []
        if self.fixed_setup:
            numbers = list(NUMBER_TOKEN_ORDER_FIXED)
        else:
            for value, count in NUMBER_TOKEN_COUNTS.items():
                numbers.extend([value] * count)
            random.shuffle(numbers)

        number_idx = 0
        for i, resource in enumerate(resources):
            coord = self.tile_coords[i]
            if resource == Resource.DESERT:
                tile = HexTile(resource=Resource.DESERT, number=None, id=i)
                tile.set_robber(True)
            else:
                tile = HexTile(resource=resource, number=numbers[number_idx], id=i)
                number_idx += 1

            tile.coord = coord
            self.tiles.append(tile)

        if len(self.tiles) != NUM_TILES:
            raise ValueError(f"Expected {NUM_TILES} tiles, got {len(self.tiles)}")

    def _create_vertices(self) -> None:
        self.vertices = [Vertex(id=i) for i in range(NUM_VERTICES)]

    def _assign_tile_vertices(self) -> None:
        """
        Deduplicate shared corners by using neighboring hex coordinates.
        """
        dirs: List[Coord] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        coord_set = set(self.tile_coords)

        vertex_key_to_id: Dict[VertexKey, int] = {}
        next_vid = 0

        for tile in self.tiles:
            tile_coord: Coord = tile.coord
            q, r = tile_coord
            tile_vertices: List[Vertex] = []

            for i in range(6):
                nbr1 = (q + dirs[i][0], r + dirs[i][1])
                nbr2 = (q + dirs[(i - 1) % 6][0], r + dirs[(i - 1) % 6][1])

                neighbors = [n for n in (nbr1, nbr2) if n in coord_set]

                if len(neighbors) == 2:
                    key = tuple(sorted([tile_coord, neighbors[0], neighbors[1]]))
                elif len(neighbors) == 1:
                    key = tuple(sorted([tile_coord, neighbors[0]]))
                else:
                    key = ("outer", tile_coord, i)

                if key not in vertex_key_to_id:
                    if next_vid >= len(self.vertices):
                        raise ValueError("Too many vertices generated")
                    vertex_key_to_id[key] = next_vid
                    self.vertices[next_vid].id = next_vid
                    next_vid += 1

                vid = vertex_key_to_id[key]
                vertex = self.vertices[vid]
                tile_vertices.append(vertex)

                if tile not in vertex.tiles:
                    vertex.tiles.append(tile)

            tile.vertices = tile_vertices

        if next_vid != NUM_VERTICES:
            raise ValueError(f"Expected {NUM_VERTICES} vertices, generated {next_vid}")

    def _create_connections(self) -> None:
        """
        Proper Catan board with 72 undirected edges.
        """
        self.connections = []
        connection_id = 0
        seen: set[tuple[int, int]] = set()

        for tile in self.tiles:
            for i in range(6):
                v1 = tile.vertices[i]
                v2 = tile.vertices[(i + 1) % 6]
                if v1.id is None or v2.id is None:
                    raise RuntimeError("Vertices must have ids before connections are created")
                edge_key = tuple(sorted((v1.id, v2.id)))

                if edge_key in seen:
                    continue
                seen.add(edge_key)

                connection = Connection(v1, v2, id=connection_id)
                self.connections.append(connection)

                v1.edges.append(connection)
                v2.edges.append(connection)

                connection_id += 1

        if len(self.connections) != NUM_CONNECTIONS:
            raise ValueError(f"Connections count must be {NUM_CONNECTIONS} (got {len(self.connections)})")

    def _assign_vertex_neighbors(self) -> None:
        for connection in self.connections:
            v1 = connection.v1
            v2 = connection.v2

            if v2 not in v1.neighbors:
                v1.neighbors.append(v2)

            if v1 not in v2.neighbors:
                v2.neighbors.append(v1)

    def _create_ports(self) -> None:
        """
        Create 9 ports and attach them to coastal vertex pairs around the shoreline.
        The exact resource ordering is still shuffled, but the attachment points now
        follow the perimeter ring instead of pairing vertices by sorted id.
        """
        port_types: List[PortType] = []
        for port_type, count in PORT_COUNTS.items():
            port_types.extend([port_type] * count)

        random.shuffle(port_types)

        port_vertex_pairs = self._coastal_port_pairs(len(port_types))

        if len(port_vertex_pairs) < len(port_types):
            raise ValueError("Not enough coastal vertex pairs to place all ports.")

        for i, port_type in enumerate(port_types):
            if port_type == PortType.GENERIC_3_TO_1:
                port = Port(resource=None, exchange_rate=3, id=i)
            elif port_type == PortType.WOOD_2_TO_1:
                port = Port(resource=Resource.WOOD, exchange_rate=2, id=i)
            elif port_type == PortType.BRICK_2_TO_1:
                port = Port(resource=Resource.BRICK, exchange_rate=2, id=i)
            elif port_type == PortType.SHEEP_2_TO_1:
                port = Port(resource=Resource.SHEEP, exchange_rate=2, id=i)
            elif port_type == PortType.WHEAT_2_TO_1:
                port = Port(resource=Resource.WHEAT, exchange_rate=2, id=i)
            elif port_type == PortType.ORE_2_TO_1:
                port = Port(resource=Resource.ORE, exchange_rate=2, id=i)
            else:
                raise ValueError(f"Unknown port type: {port_type}")

            v1, v2 = port_vertex_pairs[i]
            port.vertices = (v1, v2)
            self.ports.append(port)

            if v1.id is None or v2.id is None:
                raise RuntimeError("Port vertices must have ids")
            self.vertex_to_ports[v1.id] = port
            self.vertex_to_ports[v2.id] = port

    def _coastal_port_pairs(self, port_count: int) -> List[Tuple[Vertex, Vertex]]:
        coastal_edges: List[Connection] = []
        coastal_adjacency: Dict[int, List[int]] = {}

        for connection in self.connections:
            if len(connection.v1.tiles) >= 3 or len(connection.v2.tiles) >= 3:
                continue

            shared_tiles = {
                tile.id for tile in connection.v1.tiles if tile.id is not None
            } & {
                tile.id for tile in connection.v2.tiles if tile.id is not None
            }
            if len(shared_tiles) != 1:
                continue

            if connection.v1.id is None or connection.v2.id is None or connection.id is None:
                raise RuntimeError("Coastal edge vertices and connections must have ids")
            coastal_edges.append(connection)
            coastal_adjacency.setdefault(connection.v1.id, []).append(connection.id)
            coastal_adjacency.setdefault(connection.v2.id, []).append(connection.id)

        if not coastal_edges:
            return []

        start_edge = min(
            coastal_edges,
            key=lambda edge: (
                min(len(edge.v1.tiles), len(edge.v2.tiles)),
                min(
                    edge.v1.id if edge.v1.id is not None else -1,
                    edge.v2.id if edge.v2.id is not None else -1,
                ),
                edge.id,
            ),
        )

        ordered_edges = [start_edge]
        if start_edge.id is None or start_edge.v2.id is None:
            raise RuntimeError("Starting coastal edge must have ids")
        used_edges = {start_edge.id}
        current_vertex = start_edge.v2.id

        while len(ordered_edges) < len(coastal_edges):
            candidates = [
                edge_id
                for edge_id in coastal_adjacency.get(current_vertex, [])
                if edge_id not in used_edges
            ]
            if not candidates:
                break

            next_edge_id = min(candidates)
            next_edge = self.get_connection_by_id(next_edge_id)
            ordered_edges.append(next_edge)
            if next_edge.id is None or next_edge.v1.id is None or next_edge.v2.id is None:
                raise RuntimeError("Coastal edge traversal requires ids")
            used_edges.add(next_edge.id)
            current_vertex = next_edge.v1.id if next_edge.v2.id == current_vertex else next_edge.v2.id

        if len(ordered_edges) != len(coastal_edges):
            remaining = [edge for edge in coastal_edges if edge.id not in used_edges]
            ordered_edges.extend(sorted(remaining, key=lambda edge: edge.id if edge.id is not None else -1))

        chosen_indices = []
        total_edges = len(ordered_edges)
        for i in range(port_count):
            idx = int(round(i * total_edges / float(port_count))) % total_edges
            while idx in chosen_indices:
                idx = (idx + 1) % total_edges
            chosen_indices.append(idx)

        chosen_indices.sort()
        return [(ordered_edges[idx].v1, ordered_edges[idx].v2) for idx in chosen_indices]

    def get_tile_by_id(self, tile_id: int) -> HexTile:
        return self.tiles[tile_id]

    def get_vertex_by_id(self, vertex_id: int) -> Vertex:
        return self.vertices[vertex_id]

    def get_connection_by_id(self, connection_id: int) -> Connection:
        return self.connections[connection_id]

    def get_desert_tile(self) -> Optional[HexTile]:
        for tile in self.tiles:
            if tile.resource == Resource.DESERT:
                return tile
        return None

    def get_robber_tile(self) -> Optional[HexTile]:
        for tile in self.tiles:
            if getattr(tile, "has_robber", False):
                return tile
        return None

    def move_robber_to_tile(self, tile_id: int) -> None:
        current = self.get_robber_tile()
        if current is not None:
            current.set_robber(False)

        new_tile = self.get_tile_by_id(tile_id)
        new_tile.set_robber(True)

    def get_tiles_for_roll(self, dice_value: int) -> List[HexTile]:
        return [
            tile
            for tile in self.tiles
            if tile.number == dice_value and not getattr(tile, "has_robber", False)
        ]

    def get_vertices_adjacent_to_tile(self, tile_id: int) -> List[int]:
        tile = self.get_tile_by_id(tile_id)
        vertex_ids: List[int] = []
        for vertex in tile.vertices:
            if vertex.id is None:
                raise RuntimeError("Tile vertex must have an id")
            vertex_ids.append(vertex.id)
        return vertex_ids

    def get_player_starting_resources_from_second_settlement(self, vertex_id: int) -> Dict[Resource, int]:
        """
        Rulebook: collect one matching resource card for each hex adjacent
        to the second settlement, ignoring desert.
        """
        vertex = self.get_vertex_by_id(vertex_id)
        out = {
            Resource.WOOD: 0,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        }

        for tile in vertex.tiles:
            if tile.resource in out:
                out[tile.resource] += 1

        return out

    def get_valid_settlement_vertices(
        self,
        player_id: PlayerId,
        settlement_positions: dict[PlayerId, set],
        road_positions: dict[PlayerId, set],
        city_positions: dict[PlayerId, set] | None = None,
        require_road: bool = True,
    ) -> list[int]:
        valid: List[int] = []
        for vertex in self.vertices:
            if vertex.id is None:
                raise RuntimeError("Vertex must have an id")
            occupied = False
            for pid, positions in settlement_positions.items():
                if vertex.id in positions:
                    occupied = True
                    break
            if city_positions is not None and not occupied:
                for pid, positions in city_positions.items():
                    if vertex.id in positions:
                        occupied = True
                        break
            if occupied:
                continue

            if vertex.can_place_settlement(
                player_id,
                settlement_positions,
                road_positions,
                city_positions=city_positions,
                require_road=require_road,
            ):
                valid.append(vertex.id)
        return valid

    def get_valid_road_connections(
        self,
        player_id: PlayerId,
        settlement_positions: dict[PlayerId, set],
        road_positions: dict[PlayerId, set],
        city_positions: dict[PlayerId, set] | None = None,
    ) -> list[int]:
        valid: List[int] = []
        for conn in self.connections:
            if conn.id is None:
                raise RuntimeError("Connection must have an id")
            if conn.id in road_positions[player_id]:
                continue
            if conn.can_build_road(
                player_id,
                settlement_positions,
                road_positions,
                city_positions=city_positions,
            ):
                valid.append(conn.id)
        return valid

    def get_port_for_vertex(self, vertex_id: int) -> Optional[Port]:
        return self.vertex_to_ports.get(vertex_id)

    def __repr__(self):
        return (
            f"BoardLayout(tiles={len(self.tiles)}, "
            f"vertices={len(self.vertices)}, "
            f"connections={len(self.connections)}, "
            f"ports={len(self.ports)}, "
            f"fixed_setup={self.fixed_setup})"
        )
