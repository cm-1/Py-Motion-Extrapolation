import timeit
from enum import Enum
import random
import multiprocessing
from multiprocessing.pool import Pool, ThreadPool

class RUN_MODE(Enum):
    SINGLE = 1
    THREADS = 2
    MP = 3

def collatz(n):
    iters = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        iters += 1
    return iters

def collatz_single(x):
    return [collatz(n) for n in x]

def collatz_threaded(x, proc_count):
    result = None
    with ThreadPool(proc_count) as pool:
        result = pool.map(collatz, x)
    return result

def collatz_mp(x, proc_count):
    result = None
    with Pool(processes=proc_count) as pool:
        result = pool.map(collatz, x)
    return result

def main(x, mode, proc_count):
    if mode == RUN_MODE.SINGLE:
        return collatz_single(x)
    elif mode == RUN_MODE.THREADS:
        return collatz_threaded(x, proc_count)
    elif mode == RUN_MODE.MP:
        # raise Exception("Not working yet!")
        return collatz_mp(x, proc_count)
    else:
        raise Exception("Unsupported mode!")

# Using Pool on at least Windows requires wrapping everything in main(), 
# because each child process is a copy of the parent, but their __name__ will
# be different, so that can be used to make sure they do not recursively spawn
# their own children.
# Another multiprocessing note: haven't found anything online yet on getting it
# to work with jupyter on Windows (though I didn't spend a long time searching).
# For these reasons, joblib seems to be the way to usually go, but leaving this
# test here nonetheless.
if __name__ == "__main__":
    proc_count = multiprocessing.cpu_count()

    print("CPU_COUNT:", proc_count)
    x = [random.randint(9000000, 9999999) for _ in range(100000)]
    for mode in RUN_MODE:
        t = timeit.timeit(
            'main(x, mode, proc_count)', 
            globals=globals(), number = 5
        )
        print("time for mode", mode.name, ":", t)
