"""Guards for long unattended FEMM runs.
keep_awake()    stop Windows idle sleep while the run is alive (SetThreadExecutionState; released when the process ends)
dump_atomic()   JSON write via temp file + os.replace, so a reboot or kill mid-write never leaves a truncated cache
femm_retry()    if FEMM dies or hangs (pywintypes.com_error 'remote procedure call failed'), reopen it and redo the solve"""
import ctypes, functools, json, os, time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except (AttributeError, OSError):
        pass


def dump_atomic(obj, fn, **kw):
    tmp = fn + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, **kw); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, fn)


def femm_retry(reset, tries=3):
    """reset(): close the dead FEMM and clear the caller's 'geometry built' state so the next call rebuilds it."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **k):
            for n in range(tries):
                try:
                    return fn(*a, **k)
                except Exception as e:
                    if type(e).__name__ != 'com_error' or n == tries - 1:
                        raise
                    print('FEMM COM error (%s), restarting FEMM, retry %d' % (e, n + 1), flush=True)
                    reset(); time.sleep(5)
        return wrapped
    return deco


def kill_femm_children():
    """Close the FEMM windows of this run's pool workers. Call while the pool is still alive (inside the with block):
    once the workers exit, their FEMM windows are orphaned and no longer show up as children."""
    import psutil
    for ch in psutil.Process().children(recursive=True):
        try:
            if ch.name().lower() == 'femm.exe':
                ch.kill()
        except psutil.Error:
            pass


def _reg_file(main_pid):
    import tempfile
    return os.path.join(tempfile.gettempdir(), 'femm_pids_%d.txt' % main_pid)


def register_femm():
    """Call in a pool worker right after femm.openfemm(): records this worker's FEMM window under the parent run."""
    import psutil
    pids = [c.pid for c in psutil.Process().children() if c.name().lower() == 'femm.exe']
    with open(_reg_file(os.getppid()), 'a') as fh:
        fh.write(''.join('%d\n' % p for p in pids))


def kill_registered():
    """Close only the FEMM windows registered by this run's workers (never other runs' FEMM)."""
    import psutil
    fn = _reg_file(os.getpid())
    if not os.path.exists(fn):
        return
    for line in open(fn):
        try:
            p = psutil.Process(int(line))
            if p.name().lower() == 'femm.exe':
                p.kill()
        except (ValueError, psutil.Error):
            pass
    os.remove(fn)
