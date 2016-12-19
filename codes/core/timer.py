import time


class Timer(object):
    def __init__(self,
                 name=None):
        self.name = name
        self.reset_timer()

    def reset_timer(self):
        self.start_ = 0
        self.end_ = 0
        self.elapsed = 0

    def start(self):
        self.reset_timer()
        self.start_ = time.clock()

    def stop(self):
        self.end_ = time.clock()
        self.elapsed = self.end_ - self.start_
        assert self.elapsed >= 0, "Elapsed time can not be negative!"

    @property
    def elapsed_time(self):
        return self.elapsed

    def __str__(self):
        string = "Total time elapsed: %f secs " % self.elapsed
        if self.name:
            string += " for %s ." % self.name
        return string
