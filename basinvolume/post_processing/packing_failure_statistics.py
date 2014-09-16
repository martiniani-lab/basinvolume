from __future__ import division

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
