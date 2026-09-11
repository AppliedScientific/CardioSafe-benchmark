"""Admission control for full-data CUDA fine-tuning; no numerical training changes."""
from contextlib import contextmanager
from pathlib import Path
import json,os,sys,time

@contextmanager
def reserve_finetune_slot():
    args=sys.argv[1:]
    if not ('--model' in args and args[args.index('--model')+1]=='finetune' and
            '--device' in args and args[args.index('--device')+1]=='cuda'):
        yield;return
    import fcntl
    path=Path('/tmp/cardiosafe_revision_finetune_slots.json');lock=path.with_suffix('.lock');pid=os.getpid()
    def alive(p):
        try:
            command=Path(f'/proc/{p}/cmdline').read_bytes()
            return b'train.train_revision_neural' in command and b'finetune' in command
        except FileNotFoundError:return False
    announced=False
    while True:
        with lock.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            state=json.loads(path.read_text()) if path.exists() else {'limit':2,'pids':[]}
            state['pids']=[p for p in state['pids'] if alive(p)]
            admitted=len(state['pids'])<state['limit']
            if admitted:state['pids'].append(pid)
            path.write_text(json.dumps(state))
        if admitted:break
        if not announced:print(f"Waiting for a CUDA fine-tuning slot (limit{state['limit']})",flush=True);announced=True
        time.sleep(5)
    try:yield
    finally:
        with lock.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX);state=json.loads(path.read_text());state['pids']=[p for p in state['pids'] if p!=pid and alive(p)];path.write_text(json.dumps(state))
