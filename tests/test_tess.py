from tess import Container
from unittest import TestCase

from math import sqrt
import numpy as np

try:
    import scipy
except ImportError:
    scipy = None

import pytest
@pytest.fixture(scope="session", autouse=False)
def pytest_session_arrange():
    print("pytest_session_arrange")
    if False:
        print("\n>> TESTS SETUP")

        # check correct python env!
        import os, sys
        print("> sys.version_info", sys.version_info)
        print("> sys.executable", sys.executable)
        print("> os.getcwd()", os.getcwd())

        # redo the cython setup? cannot be done when the tests are already running...
        # but alternative calling setup + pytest from outside does not bring test explorer results
        try:
            os.system(sys.executable + " setup.py develop")
            # subprocess.call(sys.executable + " setup.py develop") # avoid invoking shell? same
        except:
            print("TESTS FAILED CYTHON RECOMPILE...")

    # this is where the testing happens
    yield
    print("\n>> TESTS DONE")

class LatticeTest:
    """A basic test for testing a lattice. The basic constants below need to be overwritten,
    as well as the setUp() function."""

    limits = None
    order = None

    def setUp(self):
        self.cells = None

    def volumes(self, vs):
        found_vs = sorted(set([round(c.volume(), 8) for c in self.cells]))
        self.assertEqual(len(vs), len(found_vs), found_vs)

        for v, fv in zip(sorted(vs), found_vs):
            self.assertAlmostEqual(v, fv)

    def neighbors(self, ns):
        found_ns = sorted(set([len(c.neighbors()) for c in self.cells]))
        self.assertEqual(len(ns), len(found_ns))

        for n, fn in zip(sorted(ns), found_ns):
            self.assertEqual(n, fn)

    def test_order(self):
        if scipy is None:
            return
        if self.order is None:
            return
        self.assertAlmostEqual(
            self.order, self.cells.order(l=6, weighted=True, local=True)
        )
        self.assertAlmostEqual(
            self.order, self.cells.order(l=6, weighted=True, local=False)
        )
        self.assertAlmostEqual(
            self.order, self.cells.order(l=6, weighted=False, local=True)
        )
        self.assertAlmostEqual(
            self.order, self.cells.order(l=6, weighted=False, local=False)
        )


class CubicLattice(LatticeTest):
    n = 4
    limits = (n, n, n)

    def get_points(self, r2=0.6):
        nm = self.n
        ptsr = [
            tuple(np.array((l, m, n)) + 0.5) + ((l + m + n) % 2,)
            for l in range(nm)
            for m in range(nm)
            for n in range(nm)
        ]
        ptsr = np.array(ptsr)
        pts = ptsr[:, :3]
        r = [0.5 if r < 1 else r2 for r in ptsr[:, 3]]
        return pts, r

    def setUp(self):
        self.pts, self.r = self.get_points(r2=0.5)
        self.cells = Container(self.pts, self.n, periodic=False)

    def test_str(self):
        self.assertEqual(str(self.cells[0]), "<Cell 0>")
        self.assertEqual(repr(self.cells[0]), "<Cell 0>")
        self.assertEqual(str(self.cells[1]), "<Cell 1>")
        self.assertEqual(repr(self.cells[1]), "<Cell 1>")
        self.assertEqual(str(self.cells[-1]), "<Cell {0}>".format(len(self.cells) - 1))
        self.assertEqual(repr(self.cells[-1]), "<Cell {0}>".format(len(self.cells) - 1))


class BasicCubic(CubicLattice, TestCase):
    order = 0.35355339059328

    def test_volumes(self):
        self.volumes([1.0])

    def test_neighbors(self):
        self.neighbors([6])

    def test_centroid(self):
        cx, cy, cz = self.cells[0].centroid()
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(cz, 0.5)

    def test_face_areas(self):
        areas = self.cells[0].face_areas()
        self.assertEqual(len(areas), 6)
        for a in areas:
            self.assertAlmostEqual(a, 1.0)

    def test_radii(self):
        for c, r in zip(self.cells, self.r):
            self.assertAlmostEqual(c.radius, 0)


class CubicAlternating(CubicLattice, TestCase):
    def setUp(self):
        self.pts, self.r = self.get_points(r2=0.6)
        self.cells = Container(self.pts, self.n, radii=self.r, periodic=True)

    def test_volumes(self):
        self.volumes([1.295031, 0.704969])

    def test_radii(self):
        for c, r in zip(self.cells, self.r):
            self.assertAlmostEqual(c.radius, r)


class CubicBlocks(CubicLattice, TestCase):
    def setUp(self):
        self.pts, self.r = self.get_points(r2=0.5)
        self.cells = Container(self.pts, self.n, periodic=False, blocks=2)

    def test_blocks(self):
        self.assertEqual(self.cells.blocks, (2, 2, 2))


class CubicBlocksUnequal(CubicLattice, TestCase):
    def setUp(self):
        self.pts, self.r = self.get_points(r2=0.5)
        self.cells = Container(self.pts, self.n, periodic=False, blocks=(0, 1.1, 2))

    def test_blocks(self):
        self.assertEqual(self.cells.blocks, (1, 1, 2))


class FCC(LatticeTest, TestCase):
    order = 0.57452425971404164
    n = 4
    limits = np.array((n, n, n))

    def points(self):
        x0 = np.array((0, 0, 0))
        x1 = np.array((0, 0.5, 0.5))
        x2 = np.array((0.5, 0, 0.5))
        x3 = np.array((0.5, 0.5, 0))

        n1, n2, n3 = self.limits
        dxs = np.array(
            [(i, j, k) for k in range(n3) for j in range(n2) for i in range(n1)]
        )
        xs = np.array([[x + dx for x in (x0, x1, x2, x3)] for dx in dxs])
        return np.concatenate(xs)

    def setUp(self):
        # also use non-origin limits
        self.cells = Container(self.points(), limits=self.limits, periodic=True)

    def test_volumes(self):
        self.volumes([0.25])

    def test_neighbors(self):
        self.neighbors([12])


class FCCelongated(FCC):
    n = 4
    limits = np.array((n, n, 2 * n))

    def test_blocks(self):
        self.assertEqual(tuple(self.cells.blocks), (6, 6, 13))


class FCCnegative(FCC):
    def setUp(self):
        # also use non-origin limits
        self.cells = Container(
            self.points(), limits=(-self.limits / 2.0, self.limits / 2.0), periodic=True
        )


class FCCmoved(FCC):
    def setUp(self):
        # also use non-origin limits
        self.cells = Container(
            self.points(),
            limits=(self.limits / 2.0, self.limits * 3.0 / 2.0),
            periodic=True,
        )


class FCCsinglelimit(FCC):
    def setUp(self):
        # also use non-origin limits
        self.cells = Container(self.points(), limits=self.n, periodic=True)


class HCP(FCC):
    order = 0.48476168522368324
    n = 4
    limits = np.array((n, n * sqrt(3.0) / 2, n * sqrt(6.0) / 3.0))

    def points(self):
        assert self.n % 2 == 0
        arr = np.array(
            [
                (
                    2 * i + (j % 2) + (k % 2),
                    sqrt(3.0) * (j + (k % 2) / 3.0),
                    sqrt(24.0) * k / 3,
                )
                for k in range(self.n)
                for j in range(self.n)
                for i in range(self.n)
            ]
        )
        return arr / 2.0

    def setUp(self):
        self.cells = Container(self.points(), limits=self.limits, periodic=True)

    def test_volumes(self):
        self.volumes([1.0 / sqrt(2.0)])

    def test_neighbors(self):
        self.neighbors([12])

class TestBoundaries(TestCase):
    limits = ((-50, -20, 80), (-30, -10, 120))

    def test_container(self):
        xs = [-31, -49]
        ys = [-11, -19]
        zs = [81, 119]

        points = [(x, y, z) for x in xs for y in ys for z in zs]
        cells1 = Container(points, limits=self.limits, periodic=False)
        self.assertEqual(len(cells1), 8)
        radii = [0.4 for p in points]
        cells2 = Container(points, limits=self.limits, radii=radii)
        self.assertEqual(len(cells2), 8)

    def test_periodic(self):
        xs = [1, 19]
        ys = [1, 9]
        zs = [1, 39]

        points = [(x, y, z) for x in xs for y in ys for z in zs]
        cells1 = Container(points, limits=self.limits, periodic=True)
        self.assertEqual(len(cells1), 8)
        radii = [0.4 for p in points]
        cells2 = Container(points, limits=self.limits, radii=radii, periodic=True)
        self.assertEqual(len(cells2), 8)

    def test_out_of_bounds(self):
        limits = [(6.03961, -2.5, 3), (9.63211, 8.66222, 8.68142)]
        points = [
            [9.63211, 0.1, 6.21613],
            [6.86854, 0.1, 5.03901],
            [8.64132, 0.1, 3],
            [6.03961, 0.1, 4.95501],
            [7.17098, 0.1, 8.68142],
            [9.63211, 3.26588, 6.21613],
            [6.86854, -0.57493, 5.03901], # skipped negative
            [8.64132, 4.43874, 3],
            [6.03961, 5.5544, 4.95501],
            [7.17098, 6.71017, 8.68142],
            [9.63211, 6.72764, 6.21613],
            [6.86854, 2.83045, 5.03901],
            [8.64132, 8.30812, 3],
            [6.03961, 8.39531, 4.95501],
            [7.17098, 8.66222, 8.68142],
        ]

        # points ON LIMIT may return point_inside true and then pruduce and exception in put()
        # so this asserts that error not really correct discard of points
        with self.assertRaises(AssertionError):
            Container(points, limits=limits)
        with self.assertRaises(AssertionError):
            Container(points, limits=limits, radii=[0.1] * len(points))

        limits = 10
        cont = Container(points, limits=limits)
        self.assertNotEqual(len(cont), len(points))
        contPoly = Container(points, limits=limits)
        self.assertNotEqual(len(contPoly), len(points))
        self.assertEqual(len(cont), len(contPoly))

    def assertListAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """" Utility function to compare a pair of lists """
        self.assertEqual(len(first), len(second), msg=msg)
        for v1, v2 in zip(first, second):
            self.assertAlmostEqual(v1, v2, places=places, msg=msg, delta=delta)

    def test_get_widths(self):
        limits = [(6, -2.5, 3), (10, 9, 9)]
        points = [
            [9.63211, 0.1, 6.21613],
            [6.86854, 0.1, 5.03901],
            [8.64132, 0.1, 3.8377],
            [6.03961, 0.1, 4.95501],
            [7.17098, 0.1, 8.68142],
            [9.63211, 3.26588, 6.21613],
            [6.86854, -0.57493, 5.03901],
            [8.64132, 4.43874, 3.8377],
            [6.03961, 5.5544, 4.95501],
            [7.17098, 6.71017, 8.68142],
            [9.63211, 6.72764, 6.21613],
            [6.86854, 2.83045, 5.03901],
            [8.64132, 8.30812, 3.8377],
            [6.03961, 8.39531, 4.95501],
            [7.17098, 8.66222, 8.68142],
        ]

        c = Container(points, limits=limits, periodic=False)
        w1, w2 = c.get_limits()
        self.assertListAlmostEqual(limits[0], w1)
        self.assertListAlmostEqual(limits[1], w2)

        c = Container(points, limits=limits, radii=[0.1] * len(points), periodic=False)
        w1, w2 = c.get_limits()
        self.assertListAlmostEqual(limits[0], w1)
        self.assertListAlmostEqual(limits[1], w2)
