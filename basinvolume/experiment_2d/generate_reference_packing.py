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
    
    """
