class HSExpReferenceGeneratePacking(object):
    """
    Runs equilibrium HS fluid with same parameters as an experimental
    image.
    
    Intended use is to import the experimental radii distribution and
    volume fraction, then sample a legal configuration of spheres from
    this and finally run an equlibrium fluid with this.
    This should then be stored in the same format as for the
    experimental input so that it can be fed in the same script for
    comparison.
    The idea is to run the same script that extracts the small packings
    from the experimental images on the output of this script.
    Therefore the output format of this script has to match the one of
    the experimental data files.
    
    Parameters
    ----------
    nr_particles : integer
        The number of particles in the "experimental image" that is
        being generated.
    nr_images : integer
        The number of "experimental images" that is generated.
        These are the fluid snapshots that should have the same format
        as the experimental data sets and which will be split into
        smaller, circular packings later.
    exp_data_set_index: integer
        Selects the experimental image / data set from which we are
        importing the radii and the volume fraction.
    exp_data_set_name_begin: string
        Beginning of the experimental data set name.
        This is something like "PackingsData_".
    """
