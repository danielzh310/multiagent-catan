import random
from typing import Dict, List

from core.constants import NUMBER_TOKEN_COUNTS, PORT_COUNTS, PlayerId, PortType, Resource
from core.connection import Connection
from core.hex_tile import HexTile
from core.port import Port
from core.vertex import Vertex


class BoardLayout:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        self.tiles = []
        self.vertices = []
        self.connections = []
        self.ports = []

        self._build_board()

    def _build_board(self):
        self._create_tiles()
        self._create_vertices()
        self._assign_tile_vertices()
        self._create_connections()
        self._assign_vertex_neighbors()
        self._create_ports()

    def _create_tiles(self):
        resources = (
            [Resource.WOOD] * 4
            + [Resource.BRICK] * 3
            + [Resource.SHEEP] * 4
            + [Resource.WHEAT] * 4
            + [Resource.ORE] * 3
            + [Resource.DESERT]
        )

        numbers = []
        for value, count in NUMBER_TOKEN_COUNTS.items():
            numbers.extend([value] * count)

        random.shuffle(resources)
        random.shuffle(numbers)

        self.tile_coords = [(q, r) for q in range(-2, 3) for r in range(-2, 3) if -2 <= q + r <= 2]

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

    def _create_vertices(self):
        self.vertices = [Vertex(id=i) for i in range(54)]

    def _create_connections(self):
        """Create connections for a proper Catan board with 72 edges."""
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

        if len(self.connections) != 72:
            raise ValueError(f"Connections count must be 72 (got {len(self.connections)})")

    def _assign_tile_vertices(self):
        dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        coord_set = set(self.tile_coords)

        vertex_key_to_id = {}
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

        if next_vid != 54:
            raise ValueError(f"Expected 54 vertices, generated {next_vid}")

    def _assign_vertex_neighbors(self):
        for connection in self.connections:
            v1 = connection.v1
            v2 = connection.v2

            if v2 not in v1.neighbors:
                v1.neighbors.append(v2)

            if v1 not in v2.neighbors:
                v2.neighbors.append(v1)

    def _create_ports(self):
        port_types = []
        for port_type, count in PORT_COUNTS.items():
            port_types.extend([port_type] * count)

        random.shuffle(port_types)

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

            self.ports.append(port)

    def get_tile_by_id(self, tile_id):
        return self.tiles[tile_id]

    def get_vertex_by_id(self, vertex_id):
        return self.vertices[vertex_id]

    def get_connection_by_id(self, connection_id):
        return self.connections[connection_id]

    def get_valid_settlement_vertices(self, player_id: PlayerId, settlement_positions: dict[PlayerId, set], road_positions: dict[PlayerId, set], require_road: bool = True) -> list[int]:
        valid = []
        for vertex in self.vertices:
            if vertex.id in settlement_positions[player_id]:
                continue  # already has settlement
            if vertex.can_place_settlement(player_id, settlement_positions, road_positions, require_road):
                valid.append(vertex.id)
        return valid

    def get_valid_road_connections(self, player_id: PlayerId, settlement_positions: dict[PlayerId, set], road_positions: dict[PlayerId, set]) -> list[int]:
        valid = []
        for conn in self.connections:
            if conn.id in road_positions[player_id]:
                continue  # already has road
            if conn.can_build_road(player_id, settlement_positions, road_positions):
                valid.append(conn.id)
        return valid

    def __repr__(self):
        return (
            f"BoardLayout(tiles={len(self.tiles)}, "
            f"vertices={len(self.vertices)}, "
            f"connections={len(self.connections)})"
        )