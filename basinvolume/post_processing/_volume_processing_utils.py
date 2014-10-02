from __future__ import division
try:
    import numpy as np
    import ConfigParser
    import os
    from basinvolume.post_processing import F_acc_Gaussian_Poly_HS_Fluid
    from basinvolume.utils import to_string
except ImportError as err:
    print err

class PackingFailureStatistics(object):
    def __init__(self, total_nr):
        self.total_nr = total_nr
        self.total_count = 0
        self.success_count = 0
    def add_success(self):
        self.add_any()
        self.success_count += 1
    def add_failure(self):
        self.add_any()
    def add_any(self):
        self.total_count += 1
    def get_nr_failures(self):
        return self.total_count - self.success_count
    def print_failure_info(self):
        print self.get_nr_failures(), "out of", self.total_count, "failed"
        print "corresponding failure ratio", self.get_nr_failures() / self.total_count
        print 100 * self.get_nr_failures() / self.total_count, "per-cent"
    def print_progress_info(self, packing_string):
        print "done", self.total_count, "out of", self.total_nr 
        print to_string(self.total_count / self.total_nr * 100, 2), "per-cent"
        print "packing was", packing_string

class VolumeSanityCheck(object):
    def __init__(self, v_acc_parameter_file):
        self.v_acc_parameter_file = v_acc_parameter_file
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.v_acc_parameter_file))
        self.nr_particles = configf.getint("PACKING", "nparticles")
        self.box_dimension = configf.getint("PACKING", "boxdim")
        self.phiHD = configf.getfloat("PACKING", "packing_fraction")
        boxv = configf.get("PACKING", "boxv")
        boxv = np.array([float(x) for x in boxv.split()])
        self.V_box = np.prod(boxv)
        self.diameter_mean = 2 * configf.getfloat("PACKING", "radii_mean")
        radii_stdev = configf.getfloat("PACKING", "radii_stdev")
        self.diameter_variance = (2 * radii_stdev) ** 2
        self.ideal_gas_V_acc = self.V_box ** self.nr_particles
        self.F0_acc = F_acc_Gaussian_Poly_HS_Fluid(self.phiHD, self.V_box, self.nr_particles, self.box_dimension, self.diameter_mean, self.diameter_variance) 
        self.V_acc = np.exp(- self.F0_acc)
        if np.log(self.V_acc) > np.log(self.ideal_gas_V_acc):
            raise Exception("VolumeSanityCheck: polyHS fluid failure")
        print "VolumeSanityCheck: "
        print "F0_acc, HS fluid", self.F0_acc
        print "F0_acc, ideal gas", - np.log(self.ideal_gas_V_acc)
    def is_insane(self, F0):
        if F0 < self.F0_acc:
            return True
        else:
            return False
    def check(self, F0, F0_name, vf_path):
        if F0 < self.F0_acc:
            print "failed F0 value", F0
            print "-log(V_acc)", self.F0_acc
            print "failed F0 name", F0_name
            print "failed packing", ([f for f in vf_path.split("/") if "jammed_packing" in f][0])[11:]
            raise Exception("VolumeSanityCheck: illegal free energy")
        if F0 < - np.log(self.ideal_gas_V_acc):
            print "failed F0 value -- failed ideal gas box test"
            print "-log(V_acc, ideal)", - np.log(self.ideal_gas_V_acc)
            print "failed F0 name", F0_name
            print "failed packing", ([f for f in vf_path.split("/") if "jammed_packing" in f][0])[11:]
            raise Exception("VolumeSanityCheck: illegal free energy")

class GLPTNotUsedStatistics(object):
    """
    Collect statistics on how many times the GL (PT) result was not used due to
    large integration error (for huge kmax).
    In these cases the analytical intrgral approximation is used.
    The number of these cases should be realtively low.
    """
    def __init__(self):
        self.total_nr = 0
        self.used_approx = 0
    def add_PT_GL(self):
        self.add_any()
    def add_approx(self):
        self.add_any()
        self.used_approx += 1
    def add_any(self):
        self.total_nr += 1
    def get_approx_use_fraction(self):
        return self.used_approx / self.total_nr
    def print_statistics(self):
        print "GLPTNotUsedStatistics:"
        print "total number of F0 values:", self.total_nr
        print "number of times GL failed:", self.used_approx
        print "GL failure (approx usage) fraction:", self.get_approx_use_fraction()
        
class BestIntegrationSelection(object):
    """
    Handles part of the analysis of F0 values.
    This version only considers packings where both, the kindk runs, and PT
    completed successfully.
    In case there is a huge kmax, we allow for the GL integration to fail and
    use an analytic approximation instead.
    In case there is no huge kmax and the GL integration still fails, something
    went wrong and we throw a warning and exception.
    In case the final F0, be it from GL integration or from the approximated
    integral, fails the constraint given by the box size, we throw a warning
    and exception.
    """
    def __init__(self, max_relative_GL_error = 0.2, kmax_threshold = 1000):
        if max_relative_GL_error < 0:
            raise Exception("BestIntegrationSelection: illegal input: max_relative_GL_error")
        self.max_relative_GL_error = max_relative_GL_error
        if kmax_threshold < 0:
            raise Exception("BestIntegrationSelection: illegal input: kmax_threshold")
        self.kmax_threshold = kmax_threshold
        self.F0_final = []
        self.F0_error_final = []
        self.bad_volumes_larger_than_Vacc = []
        self.bad_volumes_failed_GL_integration = []
        self.bad_volumes_huge_kmax = []
        self.GLPT_not_used_statistics = GLPTNotUsedStatistics()
    def check_next_F0(self, volume_sanity_check, volume_data, volume_file_path):
        F0 = volume_data.F0[-1]
        assert(F0 == F0)
        assert(np.isfinite(F0))
        F0_error = volume_data.sigF0[-1]
        assert(F0_error == F0_error)
        assert(np.isfinite(F0_error))
        F0_approx_PTu2k0 = volume_data.F0_approx_PTu2k0[-1]
        assert(F0_approx_PTu2k0 == F0_approx_PTu2k0)
        assert(np.isfinite(F0_approx_PTu2k0))
        F0_approx_PTu2k0_error = volume_data.F0_approx_PTu2k0_error[-1]
        assert(F0_approx_PTu2k0_error == F0_approx_PTu2k0_error)
        assert(np.isfinite(F0_approx_PTu2k0_error))
        kmax = self.get_kmax(volume_file_path)
        fail_information = "F0 from PT:", to_string(F0, 3), "kmax:", to_string(kmax, 3), "packing_label:", ([f for f in volume_file_path.split("/") if "jammed_packing" in f][0])[11:]
        if volume_sanity_check.is_insane(F0):
            self.bad_volumes_larger_than_Vacc.append(fail_information)
        if self.kmax_is_huge(kmax):
            self.bad_volumes_huge_kmax.append(fail_information)     
        
        def record_approximation():
            self.F0_final.append(F0_approx_PTu2k0)
            self.F0_error_final.append(F0_approx_PTu2k0_error)
            self.bad_volumes_failed_GL_integration.append(fail_information)
            self.GLPT_not_used_statistics.add_approx()
        
        if volume_sanity_check.is_insane(F0):
            if self.kmax_is_huge(kmax):
                if volume_sanity_check.is_insane(F0_approx_PTu2k0) == False:
                    record_approximation()
                else:
                    print("discarded packing: GL failed, approx failed, huge kmax")
            else:
                print "GL failed, with reasonable kmax!"
                assert(False)
        else:
            if (np.abs(F0_error) / np.abs(F0)) > self.max_relative_GL_error:
                if volume_sanity_check.is_insane(F0_approx_PTu2k0) == False:
                    record_approximation()
                else:
                    print "GL failed, approx failed, with reasonable kmax!"
                    assert(False)
            else:
                #this should be the default behaviour
                self.F0_final.append(F0)
                self.F0_error_final.append(F0_error)
                self.GLPT_not_used_statistics.add_PT_GL()
                
        assert(len(self.F0_final) == len(self.F0_error_final))
        if volume_sanity_check.is_insane(self.F0_final[-1]):
            print "fail_information"
            print fail_information
            print "F0 ",F0
            print "F0_error ",F0_error
            print "kmax ",kmax
            print "F0_approx_PTu2k0 ",F0_approx_PTu2k0
            print "F0_approx_PTu2k0_error ",F0_approx_PTu2k0_error
            assert(False)
            
    def get_kmax(self, volume_file):
        path_with_kmax_info_file = os.path.split(os.path.split(volume_file)[0])[0]
        kmax_file = [path_with_kmax_info_file + "/" + f for f in os.listdir(path_with_kmax_info_file) if f.endswith(".config") and f.startswith("findk_jammed_packing")][0]
        configf = ConfigParser.ConfigParser()
        configf.read(str(kmax_file))
        return configf.getfloat("FINDK", "kmax")
    def kmax_is_huge(self, kmax):
        if kmax > self.kmax_threshold:
            return True
        else:
            return False
    def perform_sanity_check_on_final_F0(self, volume_sanity_check):
        self.GLPT_not_used_statistics.print_statistics()
        if len(self.F0_final) != len(self.F0_error_final):
            raise Exception("BestIntegrationSelection: perform_sanity_check_on_final_F0: error handling failed")
        for F0_final in self.F0_final:
            if volume_sanity_check.is_insane(F0_final):
                print F0_final
                raise Exception("BestIntegrationSelection: perform_sanity_check_on_final_F0: F0 is insane")
            
    def print_fail_information(self, packings_dir):
        def failed_to_file(path, info):
            if len(info) == 0:
                return
            f = open(path, "w")
            for line in info:
                f.write("failed packing:\n")
                for word in line:
                    f.write(str(word) + " ")
                f.write("\n")
            f.close()
        failed_to_file(packings_dir + "/bad_volumes_larger_than_Vacc", self.bad_volumes_larger_than_Vacc)
        failed_to_file(packings_dir + "/bad_volumes_failed_GL_integration", self.bad_volumes_failed_GL_integration)
        failed_to_file(packings_dir + "/bad_volumes_huge_kmax", self.bad_volumes_huge_kmax)

