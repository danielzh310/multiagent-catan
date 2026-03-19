import random

from core.constants import NUMBER_TOKEN_COUNTS, PORT_COUNTS, PortType, Resource
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
        self._create_connections()
        self._assign_tile_vertices()
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

        number_idx = 0
        for i, resource in enumerate(resources):
            if resource == Resource.DESERT:
                tile = HexTile(resource=Resource.DESERT, number=None, id=i)
                tile.set_robber(True)
            else:
                tile = HexTile(resource=resource, number=numbers[number_idx], id=i)
                number_idx += 1

            self.tiles.append(tile)

    def _create_vertices(self):
        for i in range(54):
            self.vertices.append(Vertex(id=i))

    def _create_connections(self):
        connection_id = 0

        for i in range(len(self.vertices) - 1):
            v1 = self.vertices[i]
            v2 = self.vertices[i + 1]

            connection = Connection(v1, v2, id=connection_id)
            self.connections.append(connection)

            v1.edges.append(connection)
            v2.edges.append(connection)

            connection_id += 1

    def _assign_tile_vertices(self):
        v_idx = 0

        for tile in self.tiles:
            tile.vertices = self.vertices[v_idx:v_idx + 6]

            for vertex in tile.vertices:
                vertex.tiles.append(tile)

            v_idx = (v_idx + 3) % len(self.vertices)

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

    def __repr__(self):
        return (
            f"BoardLayout(tiles={len(self.tiles)}, "
            f"vertices={len(self.vertices)}, "
            f"connections={len(self.connections)})"
        )