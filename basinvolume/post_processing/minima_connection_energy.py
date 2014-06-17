from pele.storage import Database
from pele.storage.database import Minimum

if __name__ == "__main__":
    # parameters
    fname = "file_name"
    # read in database with minima
    # need / do not have: 1. origin, 2. potential
    db = Database(fname)
    print "number of minima in database: "
    print db.number_of_minima()
    # pick origin and neighboring minimum
    origin = db.getMinimum(0)
    neighbor = db.getMinimum(1)
    # plot energy along euclidean line connecting these two