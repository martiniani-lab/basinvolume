from __future__ import print_function
import glob
import numpy as np
import os
import sys
from basinvolume.gui import HSWCASystem
from pele.gui.run import run_gui
from pele.storage import Database, Minimum
from pele.optimize._quench import modifiedfire_cpp
import pylab as pl


def quench(coords, potential, boxv, nsteps=1e6, tol=1e-9):
    res = modifiedfire_cpp(coords, potential, maxstep=(boxv[0] * 0.1), nsteps=nsteps, tol=tol)
    if not res.success:
        print("quench failed")
        return False
    return res.coords, res.energy


def merge_db(explore_dir, fname="merged_minima_list.sqlite", distinct=False):
    created_newdb = False
    for subdir, dirs, files in os.walk(explore_dir):
        for dir in dirs:
            if dir.isdigit():
                path = os.path.join(explore_dir, dir)
                file_list = glob.glob(path + "/*.sqlite")
                for file in file_list:
                    if not created_newdb:
                        system = create_system(dbname=file)
                        newdb = system.create_database(fname)
                        created_newdb = True
                    print(file)
                    db = Database(file)
                    for m in db.minima():
                        m.coords, m.energy = quench(m.coords, system.potential, system.boxv)
                        if not distinct:
                            mnew = Minimum(m.energy, m.coords)
                        else:
                            mnew = newdb.addMinimum(m.energy, m.coords)
                        try:
                            mnew.user_data.update(m.user_data)
                        except AttributeError:
                            mnew.user_data = m.user_data.copy()
                        if not distinct:
                            newdb.session.add(mnew)
    newdb.session.commit()


#    for m in newdb.minima():
#        print m.user_data


def create_system(dbname):
    print("testing whether", dbname, "exists")
    try:
        # if the database already exists get the phases
        db = Database(dbname, createdb=False)
        print(dbname, "exists.  getting parameters")
        bdim = db.get_property("bdim").value()
        eps = db.get_property("eps").value()
        sca = db.get_property("sca").value()
        boxv = db.get_property("boxv").value()
        radii = db.get_property("radii").value()
        etol = db.get_property("etol").value()
        dtol = db.get_property("dtol").value()
    except IOError:
        print(dbname, "doesn't exist")
        sys.exit(0)
    system = HSWCASystem(eps, sca, radii, boxv, bdim=bdim, dtol=dtol, etol=etol)
    return system


def run_gui_hswca(dbname=None):
    if dbname is None:
        dbname = "minima_list.sqlite"
    system = create_system(dbname=dbname)
    db = system.create_database(dbname)
    run_gui(system, db=db)


def minimum_to_value_count(m):
    try:
        count = m.user_data["count"] + 1
        return count
    except TypeError:
        return None


def minimum_to_value_k(m):
    try:
        k = m.user_data["k"] + 1
        return k
    except TypeError:
        return None


def get_minima_k_less_than(db, val):
    new_min_list = []
    for m in db.minima():
        try:
            k = m.user_data["k"]
            if k <= val:
                new_min_list.append(m)
        except TypeError:
            pass
    return new_min_list


def get_origin(db):
    for m in db.minima():
        try:
            d = m.user_data["distance"]
            if d == 0.0:
                return m
        except TypeError:
            pass


def dgraph(dbname=None):
    from PyQt4.QtGui import QApplication
    from pele.gui.ui.dgraph_dlg import DGraphDialog, reduced_db2graph

    if dbname is None:
        dbname = "minima_list.sqlite"
    system = create_system(dbname=dbname)
    db = system.create_database(dbname)
    groups = None

    app = QApplication(sys.argv)
    kwargs = {}
    kwargs["show_minima"] = False
    kwargs["order_by_energy"] = False
    kwargs["center_gmin"] = False
    kwargs["include_gmin"] = True
    kwargs["order_by_basin_size"] = True
    # kwargs["center_minimum"] = get_origin(db)
    kwargs["Emax"] = 3800
    # kwargs["nlevels"] = 20
    # kwargs["energy_function"] = get_energy
    md = DGraphDialog(db, params=kwargs)
    md.dgraph_widget._set_lineEdit("linewidth", default=0.4)
    md.rebuild_disconnectivity_graph()

    dwargs = {}
    # draw all the minima with k<=val
    dwargs["color"] = "blue"
    dwargs["marker"] = "^"
    dwargs["zorder"] = 99
    dwargs["alpha"] = 1
    minima_k0 = get_minima_k_less_than(db, 3)
    md.dgraph_widget.dg.draw_minima(minima_k0, **dwargs)
    # draw origin
    dwargs["color"] = "red"
    dwargs["marker"] = "o"
    dwargs["zorder"] = 100
    m_origin = [get_origin(db)]
    md.dgraph_widget.dg.draw_minima(m_origin, **dwargs)

    md.dgraph_widget.canvas.draw()

    #    md.dgraph_widget.dg.color_by_value(minimum_to_value_count,colormap=get_cmap('jet'))
    #    md.dgraph_widget.redraw_disconnectivity_graph()
    md.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # merge_db(os.getcwd()+'/explore_bv_jammed_packing1',distinct=True)
    # run_gui_hswca(dbname='merged_minima_list.sqlite')
    dgraph(dbname="merged_minima_list.sqlite")
    # run_gui_hswca()
