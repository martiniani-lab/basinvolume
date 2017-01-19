from __future__ import division
import numpy as np
import abc
import os
from pele.potentials import HS_WCA
from pele.storage import Minimum
from basinvolume.utils import read_xydr, read_xyzdr
from basinvolume.gui import HSWCASystem
from basinvolume.spheres.generate_jammed_packing import import_jammed_packing_config
import ConfigParser
import time
import re
import pylab
from pele.gui import run_gui
import argparse
import sys

class build_minima_list_db(object):
    """
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *target_packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    """

    def __init__(self, fname, db_path='minima_list.sqlite', packings_dir='jammed_packings', base_dir=None):
        self.fname = fname
        dname = fname
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]

        if base_dir is None:
            base_directory = os.path.join(os.getcwd(),'explore_bv_'+str(dname))
            assert(os.path.exists(base_directory))
        else:
            if not os.path.isabs(base_dir):
                base_directory = os.path.join(os.getcwd(),packings_dir)
        self.base_directory = base_directory

        if not os.path.isabs(db_path):
            db_path = os.path.join(base_directory,db_path)
        self.db_path = db_path

        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir

        self.packing_configpath = os.path.join(packings_dir,'jammed_packings.config')

        import_jammed_packing_config(self, str(self.packing_configpath))
        self._import_packing_configuration()

        self.eps = 1
        self.system = HSWCASystem(self.eps, self.sca, self.hs_radii, self.boxv, bdim=self.bdim)
        self.potential = self.system.get_potential()
        self.db = self.system.create_database(self.db_path)

    def _import_packing_configuration(self):
        """imports the coordinates, data relative to the shape of the particles and
        whether the particles are rattlers or not. Note that self.rattlers returned
        here is of size self.ndim but in generate_jammed_packings is of size self.nparticles.
        This should be run in initialise()
        """
        path = os.path.join(self.packings_dir,self.fname)
        if self.bdim == 2:
            self.coords, hs_diameters, self.rattlers = read_xydr(path)
        elif self.bdim == 3:
            self.coords, hs_diameters, self.rattlers = read_xyzdr(path)
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        self.hs_radii = hs_diameters/2

def main():
#    parser = argparse.ArgumentParser(description="analyse hard disks/spheres packings")
#    parser.add_argument("-e","--etol", type=float, help="tolerance on particles eigenvalues, if eval < etol particle will be considered a rattler",default=1.0)
#    parser.add_argument("--show", action='store_true', help="show histograms",default=False)
#    parser.add_argument("--packingsdir", type=str, help="name of directory with packings, must be in cwd", default="jammed_packings")
#    args = parser.parse_args()
#    print args

    analyse = build_minima_list_db('jammed_packing0.xyzdr')
    #run_gui(analyse.system, analyse.db)

    system = analyse.system
    db = analyse.db


    from PyQt4.QtGui import QApplication

    mindist = system.get_mindist()
    m0 = db.getMinimum(1)

    def get_energy(mts):
        """
        this function returns the distance between the origin coordinates and
        minima or transition states. If a TS
        """
        try:
            return max(get_energy(mts.minimum1), get_energy(mts.minimum2))
        except AttributeError:
            return mindist(m0.coords, mts.coords)[0]

    def get_count(m):
        try:
            count = m.user_data["count"]
            return count
        except TypeError:
            return 0.0

    def get_distance(m):
        try:
            distance = m.user_data["distance"]
            return distance
        except TypeError:
            return 0.0

    if False:
        from pele.gui.ui.dgraph_dlg import DGraphDialog, reduced_db2graph

        kwargs = {}
        groups = None

        app = QApplication(sys.argv)
        kwargs["show_minima"] = False
        kwargs["energy_function"] = get_energy
        md = DGraphDialog(db, params=kwargs)
        md.rebuild_disconnectivity_graph()
        if groups is not None:
            md.dgraph_widget.dg.color_by_group(groups)
            md.dgraph_widget.redraw_disconnectivity_graph()
        md.show()
        sys.exit(app.exec_())

    if False:
        from pele.gui.graph_viewer import GraphViewDialog
        from OpenGL.GLUT import glutInit
        app = QApplication(sys.argv)

        wnd = GraphViewDialog(db, app=app, minima_color_value=get_distance)
        #decrunner = DECRunner(system, db, min1, min2, outstream=wnd.textEdit_writer)
        glutInit()
        wnd.show()
        from PyQt4.QtCore import QTimer
        def start():
            wnd.start()

        QTimer.singleShot(10, start)
        sys.exit(app.exec_())

    if True:
        from pele.gui import run_gui
        run_gui(system, db)

#    m1, m2 = db.minima()[:2]
#
#    system.get_double_ended_connect(min1, min2, database, parallel)


if __name__ == "__main__":
    main()
