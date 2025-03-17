from joblib import Parallel, delayed, parallel_config
import joblib
from math import ceil, floor

class Calculator:
    def __init__(self, x: float):
        self.x: float = x
		
    @staticmethod
    def _collatz(i):
        iters = 0
        while i != 1:
            if i % 2 == 0:
                i = i // 2
            else:
                i = 3*i + 1
            iters += 1
        return iters
    

    def getList(self, setting: float):
        c_input = max(1, int(abs(ceil(setting))))
        c_val = self._collatz(c_input)
        retVal = [self.x * setting, self.x - setting, c_val]
        return retVal
    
    def getListsMP(self, all_settings, max_threads_per_proc = 2):
        cpu_count = joblib.cpu_count()
        results = None
        num_procs = 1
        per_proc = 1
        if cpu_count > 1:
            num_procs = max(2, cpu_count // max_threads_per_proc)
            per_proc = int(floor(cpu_count / num_procs))
        print("Using {} jobs with max {} threads each!".format(num_procs, per_proc))
        with parallel_config(backend="loky", inner_max_num_threads=per_proc):
            results = Parallel(n_jobs=num_procs)(
                delayed(self.getList)(s) for s in all_settings
            )
        return results

