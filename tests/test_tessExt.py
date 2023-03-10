from tess import Container
from unittest import TestCase

from math import sqrt
import numpy as np

try:
    import scipy
except ImportError:
    scipy = None


class TestCell(TestCase):

    def assertListAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """" Utility function to compare a pair of lists such as a vector """
        self.assertEqual(len(first), len(second), msg=msg)
        for v1, v2 in zip(first, second):
            self.assertAlmostEqual(v1, v2, places=places, msg=msg, delta=delta)

    def assertNestedListAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """" Utility function to compare list of vectors (single nesting) """
        self.assertEqual(len(first), len(second), msg=msg)
        for l1, l2 in zip(first, second):
            self.assertEqual(len(l1), len(l2), msg=msg)
            for v1, v2 in zip(l1, l2):
                self.assertAlmostEqual(v1, v2, places=places, msg=msg, delta=delta)

    def test_methods(self):
        """Simple checks for the Cell method bindings
        """
        cell_positions = [[1., 1., 1.], [2., 2., 2.]]
        cell_radii = [0.2, 0.1]

        cells = Container(
            cell_positions, radii=cell_radii, limits=(3,3,3), periodic=False
        )

        for i, cell in enumerate(cells):

            assert cell.id == i
            assert np.allclose(cell.pos, cell_positions[i])
            assert np.isclose(cell.radius, cell_radii[i])
            assert cell.volume() > 0.0
            assert cell.max_radius_squared() > 0.0
            assert cell.total_edge_distance() > 0.0
            assert cell.surface_area() > 0.0
            assert cell.number_of_faces() > 0
            assert cell.number_of_edges() > 0
            assert len(cell.centroid()) == 3
            assert len(cell.vertex_orders()) > 0
            assert len(cell.vertices()) > 0
            assert len(cell.face_areas()) > 0
            assert len(cell.face_orders()) > 0
            assert len(cell.face_freq_table()) > 0
            assert len(cell.face_vertices()) > 0
            assert len(cell.face_perimeters()) > 0
            assert len(cell.normals()) > 0
            assert len(cell.neighbors()) > 0
            assert str(cell) == repr(cell) == f"<Cell {i}>"

    def test_methods_data(self):
        l = 0.5
        c = [0,0,0]
        bb = [ [cc-l for cc in c], [cc+l for cc in c] ]
        cont = Container(points=[(0,0,0)], limits=bb)
        cell = cont[0]

        id =                   0
        pos =                  (0.0, 0.0, 0.0)
        radius =               0.0

        centroid =             (0.0, 0.0, 0.0)
        face_areas =           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        face_freq_table =      [0, 0, 0, 0, 6]
        face_orders =          [4, 4, 4, 4, 4, 4]
        face_perimeters =      [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        face_vertices =        [[1, 3, 2, 0], [1, 5, 7, 3], [1, 0, 4, 5], [2, 6, 4, 0], [2, 3, 7, 6], [4, 6, 7, 5]]
        max_radius_squared =   3.0
        neighbors =            [-5, -2, -3, -1, -4, -6]
        normals =              [(-0.0, -0.0, -1.0), (1.0, -0.0, 0.0), (0.0, -1.0, 0.0), (-1.0, -0.0, -0.0), (0.0, 1.0, -0.0), (-0.0, 0.0, 1.0)]
        number_of_edges =      12.0
        number_of_faces =      6.0
        surface_area =         6.0
        total_edge_distance =  12.0
        vertex_orders =        [3, 3, 3, 3, 3, 3, 3, 3]
        vertices =             [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        volume =               1.0

        self.assertEqual(id, cell.id)
        self.assertListAlmostEqual(pos, cell.pos)
        self.assertAlmostEqual(radius, cell.radius)

        self.assertListAlmostEqual(centroid, cell.centroid())
        self.assertListAlmostEqual(face_areas, cell.face_areas())
        self.assertListAlmostEqual(face_freq_table, cell.face_freq_table())
        self.assertListAlmostEqual(face_orders, cell.face_orders())
        self.assertListAlmostEqual(face_perimeters, cell.face_perimeters())
        self.assertNestedListAlmostEqual(face_vertices, cell.face_vertices())
        self.assertAlmostEqual(max_radius_squared, cell.max_radius_squared())
        self.assertListAlmostEqual(neighbors, cell.neighbors())
        self.assertNestedListAlmostEqual(normals, cell.normals())
        self.assertAlmostEqual(number_of_edges, cell.number_of_edges())
        self.assertAlmostEqual(number_of_faces, cell.number_of_faces())
        self.assertAlmostEqual(surface_area, cell.surface_area())
        self.assertAlmostEqual(total_edge_distance, cell.total_edge_distance())
        self.assertListAlmostEqual(vertex_orders, cell.vertex_orders())
        self.assertNestedListAlmostEqual(vertices, cell.vertices())
        self.assertAlmostEqual(volume, cell.volume())


    def test_transform(self):
        l = 0.5
        c = [0,0,0]
        bb = [ [cc-l for cc in c], [cc+l for cc in c] ]
        cont = Container(points=[(0,0,0)], limits=bb)
        cell = cont[0]

        pos =                  (0.0, 0.0, 0.0)
        centroid =             (0.0, 0.0, 0.0)
        vertices =             [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

        self.assertListAlmostEqual(pos, cell.pos)
        self.assertListAlmostEqual(centroid, cell.centroid())
        self.assertNestedListAlmostEqual(vertices, cell.vertices())

        disp = 10
        cell.translate(disp, disp, disp)

        pos = tuple(x+disp for x in pos)
        centroid = tuple(x+disp for x in centroid)
        vertices = [tuple(x+disp for x in vert) for vert in vertices]

        self.assertListAlmostEqual(pos, cell.pos)
        self.assertListAlmostEqual(centroid, cell.centroid())
        self.assertNestedListAlmostEqual(vertices, cell.vertices())
