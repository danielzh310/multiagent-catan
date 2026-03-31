from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

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

        self.tile_coords: List[Tuple[int, int]] = []
        self.vertex_to_ports: Dict[int, Port] = {}

        self._build_board()

    def _build_board(self):
        self._create_tiles()
        self._create_vertices()
        self._assign_tile_vertices()
        self._create_connections()
        self._assign_vertex_neighbors()
        self._create_ports()

    def _create_tiles(self):
        resources = []
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

        numbers = []
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

    def _create_vertices(self):
        self.vertices = [Vertex(id=i) for i in range(NUM_VERTICES)]

    def _assign_tile_vertices(self):
        """
        Deduplicate shared corners by using neighboring hex coordinates.
        """
        dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        coord_set = set(self.tile_coords)

        vertex_key_to_id: Dict[tuple, int] = {}
        next_vid = 0

        for tile in self.tiles:
            q, r = tile.coord
            tile_vertices = []

            for i in range(6):
                nbr1 = (q + dirs[i][0], r + dirs[i][1])
                nbr2 = (q + dirs[(i - 1) % 6][0], r + dirs[(i - 1) % 6][1])

                neighbors = [n for n in (nbr1, nbr2) if n in coord_set]

                if len(neighbors) == 2:
                    key = tuple(sorted([tile.coord, neighbors[0], neighbors[1]]))
                elif len(neighbors) == 1:
                    key = tuple(sorted([tile.coord, neighbors[0]]))
                else:
                    key = ("outer", tile.coord, i)

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

    def _create_connections(self):
        """
        Proper Catan board with 72 undirected edges.
        """
        self.connections = []
        connection_id = 0
        seen = set()

        for tile in self.tiles:
            for i in range(6):
                v1 = tile.vertices[i]
                v2 = tile.vertices[(i + 1) % 6]
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

    def _assign_vertex_neighbors(self):
        for connection in self.connections:
            v1 = connection.v1
            v2 = connection.v2

            if v2 not in v1.neighbors:
                v1.neighbors.append(v2)

            if v1 not in v2.neighbors:
                v2.neighbors.append(v1)

    def _create_ports(self):
        """
        Create 9 ports and attach them to 18 coastal vertices.
        This is still a simplified port placement layer, but it preserves:
        - 4 generic 3:1 ports
        - 5 specialized 2:1 ports
        """
        port_types = []
        for port_type, count in PORT_COUNTS.items():
            port_types.extend([port_type] * count)

        random.shuffle(port_types)

        coastal_vertices = [v for v in self.vertices if len(v.tiles) < 3]
        coastal_vertices = sorted(coastal_vertices, key=lambda v: v.id)

        # Attach each port to a consecutive pair of coastal vertices.
        # This is a simplification, but good enough until you wire the exact
        # official coastal slots from the board diagram.
        port_vertex_pairs = []
        i = 0
        while i + 1 < len(coastal_vertices) and len(port_vertex_pairs) < len(port_types):
            v1 = coastal_vertices[i]
            v2 = coastal_vertices[i + 1]
            port_vertex_pairs.append((v1, v2))
            i += 2

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

            self.vertex_to_ports[v1.id] = port
            self.vertex_to_ports[v2.id] = port

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
        return [vertex.id for vertex in tile.vertices]

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
        valid = []
        for vertex in self.vertices:
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
        valid = []
        for conn in self.connections:
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