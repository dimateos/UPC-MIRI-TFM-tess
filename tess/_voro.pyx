# distutils: language = c++
# distutils: include_dirs = src
# distutils: sources = src/voro++.cc
# cython: language_level=3, boundscheck=False

from __future__ import division

from libcpp.vector cimport vector
from libcpp cimport bool as cbool
from cython.operator cimport dereference

cdef extern from "voro++.hh" namespace "voro":
    cdef cppclass container_base:
        # declared but derived classes not actually deriving
        # atm repeating the declaration of total_particles, add_wall...
        pass

    cdef cppclass wall:
        # same as container_base, atm wall_plane not actually derived from this
        pass

    cdef cppclass container:
        double ax, ay, az, bx, by, bz
        container(double,double,double,double,double,double,
                int,int,int,cbool,cbool,cbool,int) except +
        cbool compute_cell(voronoicell_neighbor &c,c_loop_all &vl)
        cbool point_inside(double,double,double)
        cbool put(int, double, double, double)

        int total_particles()
        # void add_wall(wall &w)
        void add_wall(wall *w)

    cdef cppclass container_poly:
        double ax, ay, az, bx, by, bz
        container_poly(double,double,double,double,double,double,
                int,int,int,cbool,cbool,cbool,int) except +
        cbool compute_cell(voronoicell_neighbor &c, c_loop_all &vl)
        cbool point_inside(double,double,double)
        cbool put(int, double, double, double, double)

        int total_particles()
        # void add_wall(wall &w)
        void add_wall(wall *w)


    cdef cppclass voronoicell_neighbor:
        voronoicell()
        void centroid(double &cx, double &cy, double &cz)
        double volume()
        double max_radius_squared()
        double total_edge_distance()
        double surface_area()
        double number_of_faces()
        double number_of_edges()

        void vertex_orders(vector[int] &)
        void vertices(double,double,double, vector[double]&)
        void vertices(vector[double]&)
        void face_areas(vector[double] &)
        void face_orders(vector[int] &)
        void face_freq_table(vector[int] &)
        void face_vertices(vector[int] &)
        void face_perimeters(vector[double] &)
        void normals(vector[double] &)
        void neighbors(vector[int] &)

        # void translate(double,double,double)
        # cbool plane(double,double,double, double rsq)
        cbool nplane(double,double,double, double rsq, int p_id)
        # cbool plane(double,double,double)
        cbool nplane(double,double,double, int p_id)

    cdef cppclass c_loop_all:
        c_loop_all(container_base&)
        cbool start()
        cbool inc()
        int pid()
        void pos(double &x, double &y, double &z)
        void pos(int &pid, double &x, double &y, double &z, double &r)

    cdef cppclass wall_plane:
        int w_id
        double xc, yc, zc, ac
        wall_plane(double xc, double yc, double zc, double ac, int w_id)

cdef class Cell:
    """A basic voronoi cell, usually created by :class:`Container`.

    A Voronoi cell has polygonal `faces`, connected by `edges` and `vertices`.

    The various methods of a `Cell` allow access to the geometry and neighbor information."""
    cdef voronoicell_neighbor *thisptr
    cdef int _id
    cdef double x,y,z
    cdef double r

    def __cinit__(self):
        self.thisptr = new voronoicell_neighbor()

    def __dealloc__(self):
        del self.thisptr

    @property
    def pos(self):
        "The position of the initial point around which this cell was created."
        return (self.x, self.y, self.z)

    @property
    def radius(self):
        """The radius of the particle around which this cell was created.

        Defaults to 0."""
        return self.r

    @property
    def id(self):
        """The ``id`` of the cell, which should generally correspond to its index in the
        ``Container``."""
        return self._id


    def volume(self):
        "Cell volume"
        return self.thisptr.volume()
    def max_radius_squared(self):
        """Maximum distance from ``pos()`` to outer edge of the cell (I think, see ``voro++`` documentation.)"""
        return self.thisptr.max_radius_squared()
    def total_edge_distance(self):
        return self.thisptr.total_edge_distance()
    def surface_area(self):
        return self.thisptr.surface_area()
    def number_of_faces(self):
        return self.thisptr.number_of_faces()
    def number_of_edges(self):
        return self.thisptr.number_of_edges()

    def centroid(self):
            cdef double cx = 0
            cdef double cy = 0
            cdef double cz = 0
            self.thisptr.centroid(cx,cy,cz)
            x,y,z = self.pos
            return (cx+x,cy+y,cz+z)
    def centroid_local(self):
        cdef double cx = 0
        cdef double cy = 0
        cdef double cz = 0
        self.thisptr.centroid(cx,cy,cz)
        return (cx,cy,cz)

    def vertex_orders(self):
        cdef vector[int] v
        self.thisptr.vertex_orders(v)
        return v

    def vertices(self):
        """A list of all the locations of the vertices of each face.

        Returns
        -------
        A list of 3-tuples of floats. Each tuple corresponds to a single vertex."""
        cdef vector[double] v
        self.thisptr.vertices(self.x, self.y, self.z, v)
        return list(zip(v[::3], v[1::3], v[2::3]))

    def vertices_local(self):
        """A list of all the locations of the vertices of each face in local coordinates.

        Returns
        -------
        A list of 3-tuples of floats. Each tuple corresponds to a single vertex."""
        cdef vector[double] v
        self.thisptr.vertices(v)
        return list(zip(v[::3], v[1::3], v[2::3]))

    def face_areas(self):
        """A list of the areas of each face.

        Returns
        -------
        A list of floats. Each inner list corresponds to a face."""
        cdef vector[double] v
        self.thisptr.face_areas(v)
        return v

    def face_orders(self):
        """A list of the number of edges per face.

        Returns
        -------
        A list of integers. Each integer corresponds to the number of edges of the respective face."""
        cdef vector[int] v
        self.thisptr.face_orders(v)
        return v

    def face_freq_table(self):
        cdef vector[int] v
        self.thisptr.face_freq_table(v)
        return v

    def face_vertices(self):
        """A list of the indices of the vertices of each face.

        Returns
        -------
        A list of lists of ints. Each inner list corresponds to a face, and each index corresponds
        to a vertex from :meth:`vertices`.
        """

        cdef vector[int] v
        self.thisptr.face_vertices(v)

        mylist = []

        it = iter(v)
        while True:
            try:
                n = next(it)
            except StopIteration:
                break
            mylist.append([next(it) for _ in range(n)])

        return mylist

    def face_perimeters(self):
        cdef vector[double] v
        self.thisptr.face_perimeters(v)
        return v

    def normals(self):
        r"""A list of the areas of each face.

        Returns
        -------
        A list of 3-tuples of floats. Each tuple corresponds to a face."""
        cdef vector[double] v
        self.thisptr.normals(v)
        return list(zip(v[::3], v[1::3], v[2::3]))

    def neighbors(self):
        r"""
        Return a list of the *neighbors* of the current `Cell`.

        This is a list of indices, which correspond to the input points. The exception to this
        is the walls: walls are numbered -1 to -6, so an index less than 0 in the list of
        `neighbors()` indicates that a `Cell` is neighbors with a wall.
        """
        cdef vector[int] v
        self.thisptr.neighbors(v)
        return v


    def translate(self, x,y,z):
        self.x+= x # all vertices, centroid, etc are given relative to the coordinate system of the particle
        self.y+= y #    which keeps its center at the oiginal particle point
        self.z+= z
        # self.thisptr.translate(x,y,z) # translates only the vertices, plus breaks other methods!


    def cut_plane(self, nx,ny,nz, n_lenSq, p_id=0):
        """ Cut the cell by a plane defined by a normal vector xyz and its modulus squared (non-normalized distance)
            * the cut face will have reference wall id==p_id as neighbour
            * should be false when the plane removes the whole volume, but seems like it does not! instead it does nothing to the cell
        """
        # voro++ forces to cut by particle bisector, so double the distance to force the plane to cut at the expected distance
        assert self.thisptr.nplane(nx,ny,nz, 2.0*n_lenSq, p_id)

    def cut_plane_particle(self, px,py,pz, p_id=0):
        """ Cuts the cell by the plane corresponding to the perpendicular bisector of a particle positioned at xyz
            * the cut face will have reference wall id==p_id as neighbour
            * should be false likewise cut_plane, but in this case it is actually trigered by other particles?
        """
        assert self.thisptr.nplane(px,py,pz, p_id)


    def __str__(self):
        return '<Cell {0}>'.format(self._id)

    def __repr__(self):
        return '<Cell {0}>'.format(self._id)


cdef class Container:
    cdef container *thisptr
    def __cinit__(self, double ax_,double bx_,double ay_,double by_,double az_,double bz_,
                int nx_,int ny_,int nz_,cbool xperiodic_,cbool yperiodic_,cbool zperiodic_,int init_mem):
        self.thisptr = new container(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_,
                xperiodic_, yperiodic_, zperiodic_, init_mem)

    def __dealloc__(self):
        del self.thisptr

    def point_inside(self, double x, double y, double z):
        return self.thisptr.point_inside(x, y, z)

    def put(self, int n, double x, double y, double z):
        assert self.thisptr.point_inside(x, y, z)
        assert self.thisptr.put(n, x, y, z)

    def add_wall(self, double xc_, double yc_, double zc_, double ac_, int w_id_=-10):
        # only negative id, and over -6 reserved for bounding box walls -> -10 for rounding
        assert w_id_ <= -10

        # atm no inheritance?
        cdef wall_plane *wall_ptr = new wall_plane(xc_, yc_, zc_, ac_, w_id_)
        cdef wall *wall_baseptr = (<wall *>(wall_ptr))

        self.thisptr.add_wall(wall_baseptr)
        pass

    def get_cells(self):
        cdef container_base *baseptr = (<container_base *>(self.thisptr))
        cdef c_loop_all *vl = new c_loop_all(dereference(baseptr))

        cell = Cell()

        cdef int vcells_left = self.thisptr.total_particles()
        cdef int id

        mylist = [None for _ in range(vcells_left)]

        if not vl.start():
            del vl
            raise ValueError("Failed to start loop")

        while True:
            if(self.thisptr.compute_cell(dereference(cell.thisptr), dereference(vl))):
                cell._id = vl.pid()
                assert cell._id < self.thisptr.total_particles(), (
                    "Cell id %s larger than total %s" % (cell._id, self.thisptr.total_particles()))

                vl.pos(cell.x,cell.y,cell.z)
                cell.r = 0
                mylist[cell._id] = cell

                vcells_left -= 1
                cell = Cell()
            if not vl.inc(): break

        del vl

        if vcells_left != 0:
            raise ValueError("Computation failed")
        return mylist

    def get_limits(self):
        return (
            (self.thisptr.ax, self.thisptr.ay, self.thisptr.az),
            (self.thisptr.bx, self.thisptr.by, self.thisptr.bz),
        )


# Same as container but with the addition of variable radii, just duplicate the class for performance
cdef class ContainerPoly:
    cdef container_poly *thisptr
    def __cinit__(self, double ax_,double bx_,double ay_,double by_,double az_,double bz_,
                int nx_,int ny_,int nz_,cbool xperiodic_,cbool yperiodic_,cbool zperiodic_,int init_mem):
        self.thisptr = new container_poly(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_,
                xperiodic_, yperiodic_, zperiodic_, init_mem)

    def __dealloc__(self):
        del self.thisptr

    def point_inside(self, double x, double y, double z):
        return self.thisptr.point_inside(x, y, z)

    def put(self, int n, double x, double y, double z, double r):
        assert self.thisptr.point_inside(x, y, z)
        assert self.thisptr.put(n,x,y,z,r)

    def add_wall(self, double xc_, double yc_, double zc_, double ac_, int w_id_=-10):
        # only negative id, and over -6 reserved for bounding box walls -> -10 for rounding
        assert w_id_ <= -10

        # atm no inheritance, so repeat code?
        cdef wall_plane *wall_ptr = new wall_plane(xc_, yc_, zc_, ac_, w_id_)
        cdef wall *wall_baseptr = (<wall *>(wall_ptr))

        self.thisptr.add_wall(wall_baseptr)

    def get_cells(self):
        cdef container_base *baseptr = (<container_base *>(self.thisptr))
        cdef c_loop_all *vl = new c_loop_all(dereference(baseptr))

        cell = Cell()

        cdef int vcells_left = self.thisptr.total_particles()
        cdef int id

        mylist = [None for _ in range(vcells_left)]

        if not vl.start():
            del vl
            raise ValueError("Failed to start loop")

        while True:
            if(self.thisptr.compute_cell(dereference(cell.thisptr), dereference(vl))):
                vl.pos(cell._id, cell.x,cell.y,cell.z,cell.r)
                assert cell._id < self.thisptr.total_particles(), (
                    "Cell id %s larger than total %s" % (cell._id, self.thisptr.total_particles()))
                mylist[cell._id] = cell

                vcells_left -= 1
                cell = Cell()
            if not vl.inc(): break

        del vl

        if vcells_left != 0:
            raise ValueError("Computation failed")
        return mylist

    def get_limits(self):
        return (
            (self.thisptr.ax, self.thisptr.ay, self.thisptr.az),
            (self.thisptr.bx, self.thisptr.by, self.thisptr.bz),
        )
