import numpy as np
import sys
from pele.storage import Database
from basinvolume.gui import HSWCASystem
from pele.gui.run import run_gui

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

run_gui_hswca()