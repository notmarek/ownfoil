"""Library files kept inside RAR archives, read and served without ever being extracted.

A RAR is indexed by its first volume and stands for the one NSP/NSZ/XCI/XCZ it carries: the
Files row keeps the archive as `filepath` (what exists on disk, what the watcher reports) and
the member as `filename`/`extension`/`size` (what clients are offered). Nothing is unpacked to
disk - bytes come off `unrar p` through rarfile, straight into the reader or the response.

Store-mode (-m0) members read by real seeks. A compressed member can only be read forward:
reaching an offset means decompressing everything before it, and going back means starting
over. Two layers keep that affordable:

- `StreamPool` parks a download's decompressor when its response ends, still positioned, so
  a client fetching a file in consecutive ranges continues one stream instead of restarting
  it per range.
- `CachedReader` keeps recently read blocks, so the header walk identification does - which
  revisits partition tables and NCA headers - mostly hits memory instead of rewinding.
"""
import logging
import os
import threading
import time
from collections import OrderedDict

import rarfile

from utils import is_archive_path, is_container_name  # noqa: F401 - is_archive_path re-exported

logger = logging.getLogger('main')

# rarfile copies a small compressed member into a temporary single-volume archive before
# handing it to unrar. Nothing here is ever to be written to disk, so always pipe instead.
rarfile.USE_EXTRACT_HACK = False

# How much one response pulls from the decompressor at a time.
CHUNK_SIZE = 1 << 20


def tool_available():
    """Whether rarfile found a tool that can decompress (unrar, unar, 7z or bsdtar)."""
    try:
        rarfile.tool_setup()
        return True
    except rarfile.RarCannotExec:
        return False


def _member_info(rf):
    """The RarInfo of the library container an open archive carries."""
    for info in rf.infolist():
        if not info.is_dir() and is_container_name(info.filename):
            return info
    raise ValueError('Archive holds no NSP, NSZ, XCI or XCZ.')


def member(path):
    """(name, size) of the library container inside a RAR, read off the volume headers."""
    with rarfile.RarFile(path) as rf:
        info = _member_info(rf)
        if info.needs_password():
            raise ValueError('Archive is encrypted.')
        return os.path.basename(info.filename.replace('\\', '/')), info.file_size


def volumes(path):
    """Every volume file of the set `path` opens, first included."""
    with rarfile.RarFile(path) as rf:
        return rf.volumelist()


class MemberReader:
    """A forward-preferring file object over an archive member.

    Seeking forward decompresses and discards, seeking back restarts the decompressor;
    rarfile turns both into real seeks when the member is stored.
    """

    def __init__(self, path):
        self.path = path
        self._rf = rarfile.RarFile(path)
        try:
            info = _member_info(self._rf)
            self.size = info.file_size
            self._f = self._rf.open(info)
        except Exception:
            self._rf.close()
            raise

    def tell(self):
        return self._f.tell()

    def seek(self, offset, whence=0):
        return self._f.seek(offset, whence)

    def read(self, size=-1):
        return self._f.read(size)

    def flush(self):
        pass

    def close(self):
        try:
            self._f.close()
        finally:
            self._rf.close()


class CachedReader:
    """A MemberReader behind an LRU of fixed-size blocks.

    nsz reads a container in small pieces and keeps returning to the tables it read before;
    through a bare MemberReader each such return would restart decompression from byte 0.
    """

    def __init__(self, reader, block_size=1 << 20, max_blocks=64):
        self._reader = reader
        self.size = reader.size
        self._block_size = block_size
        self._max_blocks = max_blocks
        self._blocks = OrderedDict()
        self._pos = 0

    def _block(self, index):
        block = self._blocks.get(index)
        if block is not None:
            self._blocks.move_to_end(index)
            return block
        start = index * self._block_size
        if self._reader.tell() != start:
            self._reader.seek(start)
        block = self._reader.read(self._block_size)
        self._blocks[index] = block
        if len(self._blocks) > self._max_blocks:
            self._blocks.popitem(last=False)
        return block

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        if whence == 1:
            offset += self._pos
        elif whence == 2:
            offset += self.size
        self._pos = max(0, min(offset, self.size))
        return self._pos

    def read(self, size=-1):
        end = self.size if size is None or size < 0 else min(self.size, self._pos + size)
        out = bytearray()
        while self._pos < end:
            index, within = divmod(self._pos, self._block_size)
            block = self._block(index)
            piece = block[within:within + end - self._pos]
            if not piece:
                break
            out += piece
            self._pos += len(piece)
        return bytes(out)

    def flush(self):
        pass

    def close(self):
        self._blocks.clear()
        self._reader.close()


class StreamPool:
    """Idle, still-positioned member readers, so a follow-up range resumes where one ended.

    A reader is checked out by one response at a time. Taking one hands back the idle reader
    for that archive furthest along without passing the wanted offset - skipping forward from
    it never costs more than starting over. Readers idle past `ttl`, or beyond `max_idle` in
    all, are closed.
    """

    def __init__(self, max_idle=8, ttl=120.0, opener=MemberReader):
        self._lock = threading.Lock()
        self._idle = []  # [(path, reader, parked_at)]
        self._max_idle = max_idle
        self._ttl = ttl
        self._opener = opener
        self._reaper = None

    def take(self, path, offset):
        with self._lock:
            stale = self._expire_locked()
            best = None
            for i, (p, reader, _) in enumerate(self._idle):
                if p != path:
                    continue
                pos = reader.tell()
                if pos <= offset and (best is None or pos > self._idle[best][1].tell()):
                    best = i
            reader = self._idle.pop(best)[1] if best is not None else None
        for r in stale:
            _close_quietly(r)
        if reader is None:
            reader = self._opener(path)
        if reader.tell() != offset:
            try:
                reader.seek(offset)
            except Exception:
                _close_quietly(reader)
                raise
        return reader

    def give(self, path, reader):
        if reader.tell() >= reader.size:
            # Nothing follows the end of a member: no later range can resume this one.
            _close_quietly(reader)
            return
        stale = []
        with self._lock:
            self._idle.append((path, reader, time.monotonic()))
            stale += self._expire_locked()
            while len(self._idle) > self._max_idle:
                stale.append(self._idle.pop(0)[1])
            self._arm_reaper_locked()
        for r in stale:
            _close_quietly(r)

    def clear(self):
        with self._lock:
            idle, self._idle = self._idle, []
        for _, reader, _ in idle:
            _close_quietly(reader)

    def _expire_locked(self):
        now = time.monotonic()
        keep, stale = [], []
        for entry in self._idle:
            (keep if now - entry[2] < self._ttl else stale).append(entry)
        self._idle = keep
        return [reader for _, reader, _ in stale]

    def _arm_reaper_locked(self):
        # Idle readers each hold an unrar process; without traffic nothing else would end them.
        if self._reaper is not None or not self._idle:
            return
        self._reaper = threading.Timer(self._ttl, self._reap)
        self._reaper.daemon = True
        self._reaper.start()

    def _reap(self):
        with self._lock:
            self._reaper = None
            stale = self._expire_locked()
            self._arm_reaper_locked()
        for r in stale:
            _close_quietly(r)


def _close_quietly(reader):
    try:
        reader.close()
    except Exception as e:
        logger.debug(f'Error closing archive reader for {getattr(reader, "path", "?")}: {e}')


pool = StreamPool()


def stream(path, start, stop):
    """Yield bytes [start, stop) of the member in `path`, parking the reader when done.

    The reader is taken on first iteration, so a response whose body is never sent (HEAD)
    starts no decompressor. A client that disconnects mid-range still parks its reader at
    wherever it got to - its retry is usually a range starting right there.
    """
    reader = pool.take(path, start)
    healthy = False
    try:
        remaining = stop - start
        while remaining > 0:
            data = reader.read(min(CHUNK_SIZE, remaining))
            if not data:
                raise IOError(f'Archive member ended early: {path}')
            remaining -= len(data)
            yield data
        healthy = True
    except GeneratorExit:
        healthy = True
        raise
    finally:
        if healthy:
            pool.give(path, reader)
        else:
            _close_quietly(reader)
