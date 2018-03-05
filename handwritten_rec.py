from utility import *
import numpy as np
from time import sleep
from multiprocessing import Pool


class Deco:
    def __init__(self, pids):
        self.pids = pids

    def __call__(self, haha):
        return do_cut(haha, self.pids)


def do_cut(haha, pids):
    if os.getpid() not in pids:
        pids.append(os.getpid())
    position = np.where(pids == os.getpid())[0]
    if not position:
        position = 0

    did = 0
    for cut in tqdm.tqdm(range(100), desc='{0:s}, {1:d}, {2:d}'.format(did, position, os.getpid()), position=position, leave=False):
        sleep(0.1)
        did += 1


if __name__ == '__main__':
    from numpy import memmap, uint64

    pids = memmap(os.path.join(tempfile.mkdtemp(), 'test'), dtype=uint64, shape=num_cores, mode='w+')
    results = Parallel(n_jobs=num_cores)(delayed(utils.do_cut)(......, pids) for task in tasks)

    pool = Pool(2)
    pids = []
    deco_do_cut = Deco(pids)
    lst = list(range(5))
    map_result = pool.map_async(deco_do_cut, lst)
    pool.close()
    pool.join()
    print(map_result)
