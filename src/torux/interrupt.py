"""Ctrl-C during compiled work: held while a compiled call runs and raised as a plain KeyboardInterrupt once it returns.

    with held() as asked:
        for chunk in chunks:            # each a second or so of compiled work
            kernel(chunk)
            if asked():
                break                   # leaving the block raises KeyboardInterrupt

Python's own handler would raise inside the compiled call, where numba either finishes the whole call first or ends
it in a SystemError that no ``except KeyboardInterrupt`` catches.  So while the block runs SIGINT only notes that it
came, the work checks between its chunks, and the block raises KeyboardInterrupt as it ends.  Where Python's handler
is not the one in place (a thread other than the main one, Ctrl-C ignored, a handler of the caller's own) nothing is
changed and asked() is always False.
"""
from __future__ import annotations

import contextlib
import signal
import threading


@contextlib.contextmanager
def held():
    if threading.current_thread() is not threading.main_thread() \
            or signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
        yield lambda: False
        return
    came = []
    signal.signal(signal.SIGINT, lambda signum, frame: came.append(signum))
    try:
        yield lambda: bool(came)
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    if came:
        raise KeyboardInterrupt
