from basinvolume.gui import HSWCASystem
from pele.gui.run import run_gui
from pele.storage import Database, Minimum
import glob
import numpy as np
import os
import sys

def merge_db(explore_dir, fname='merged_minima_list.sqlite', distinct=False):
    created_newdb=False 
    for subdir, dirs, files in os.walk(explore_dir):
        for dir in dirs:
            if dir.isdigit():
                path = os.path.join(explore_dir,dir)
                file_list = glob.glob(path + '/*.sqlite')
                for file in file_list:
                    if not created_newdb:
                        system = create_system(dbname=file)
                        newdb = system.create_database(fname)
                        created_newdb=True
                    print file
                    db = Database(file)
                    for m in db.minima():
                        if not distinct:
                            mnew = Minimum(m.energy,m.coords)
                        else:
                            mnew = newdb.addMinimum(m.energy,m.coords)
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
    print "testing whether", dbname, "exists"
    try:
        # if the database already exists get the phases
        db = Database(dbname, createdb=False)
        print dbname, "exists.  getting parameters"
        bdim = db.get_property("bdim").value()
        eps = db.get_property("eps").value()
        sca = db.get_property("sca").value()
        boxv = db.get_property("boxv").value()
        radii = db.get_property("radii").value()
        etol = db.get_property("etol").value()
        dtol = db.get_property("dtol").value()
    except IOError:
        print dbname, "doesn't exist"
        sys.exit(0)
    system = HSWCASystem(eps, sca, radii, boxv, 
                         bdim=bdim, dtol=dtol, etol=etol)
    return system

def run_gui_hswca(dbname=None):
    if dbname is None:
        dbname = "minima_list.sqlite"
    system = create_system(dbname=dbname)
    db = system.create_database(dbname)
    run_gui(system, db=db)

def minimum_to_value(m):
    try:
        count = m.user_data["count"]
        return count
    except TypeError:
        return None  

def dgraph(dbname=None):
    from PyQt4.QtGui import QApplication
    from pele.gui.ui.dgraph_dlg import DGraphDialog, reduced_db2graph
    
    if dbname is None:
        dbname = "minima_list.sqlite"
    system = create_system(dbname=dbname)
    db = system.create_database(dbname)
    kwargs = {}
    groups = None
    
    app = QApplication(sys.argv) 
    kwargs["show_minima"] = True
    #kwargs["energy_function"] = get_energy
    md = DGraphDialog(db, params=kwargs)
    md.rebuild_disconnectivity_graph()
    md.dgraph_widget.dg.color_by_value(minimum_to_value)
    md.dgraph_widget.redraw_disconnectivity_graph()
    md.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    #merge_db(os.getcwd()+'/explore_bv_jammed_packing184',distinct=False)
    run_gui_hswca(dbname='merged_minima_list.sqlite')
    #dgraph(dbname='merged_minima_list.sqlite')
    #run_gui_hswca()