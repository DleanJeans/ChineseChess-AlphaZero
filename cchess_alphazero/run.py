import os
import sys
import multiprocessing as mp

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

def main():
    # mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)
    from cchess_alphazero import manager
    manager.start()

if __name__ == "__main__":
    main()