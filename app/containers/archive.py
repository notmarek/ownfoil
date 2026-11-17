"""Library files kept inside RAR archives, read and served without extracting them in place.

A RAR is indexed by its first volume and stands for the one NSP/NSZ/XCI/XCZ it carries: the
Files row keeps the archive as `filepath` (what exists on disk, what the watcher reports) and
the member as `filename`/`extension`/`size` (what clients are offered). Bytes come off
`unrar p` through rarfile.

A compressed member can only be read forward: reaching an offset means decompressing
everything before it. That decides how each consumer reads:

- Identification reads through `CachedReader`, which keeps recently read blocks: the header
  walk revisits partition tables and NCA headers, and those mostly hit memory.
- Downloads are served out of an `Extraction`: the member unpacked at full speed into an
  anonymous temporary file, shared by every request for it. Installers read out of order and
  give up on a read that stalls (Sphaira after 15s), so a request has to find the data
  already unpacked, or the unpacking well on its way to it, instead of starting over.
"""
import logging
import os
import threading
import tempfile
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


def _cache_dir():
    """Where unpacked members live while served. Point it at a tmpfs to keep them in memory."""
    return os.environ.get('OWNFOIL_ARCHIVE_TMP') or tempfile.gettempdir()


def _cache_limit():
    """Bytes the unpacked members may take together, from OWNFOIL_ARCHIVE_TMP_MAX_GB; 0 = no cap."""
    try:
        return int(float(os.environ.get('OWNFOIL_ARCHIVE_TMP_MAX_GB', '0')) * (1 << 30))
    except ValueError:
        return 0


class Extraction:
    """One member being unpacked, at full speed, into an anonymous temporary file.

    Reads are served out of the file as soon as the unpacking has got past them, so a client
    jumping ahead waits for the unpacker instead of starting a decompressor of its own, and a
    client retrying a read that timed out finds it further along. The file has no name on
    disk (POSIX unlinks it at creation, Windows on close), so nothing outlives the process.
    """

    def __init__(self, path, opener=MemberReader):
        self.path = path
        self._opener = opener
        self._cond = threading.Condition()
        self._file_lock = threading.Lock()
        self._file = tempfile.TemporaryFile(dir=_cache_dir(), prefix='ownfoil-')
        self.size = None
        self.done = 0
        self.error = None
        self.users = 0
        self.last_used = time.monotonic()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=f'unrar {os.path.basename(path)}',
                                        daemon=True)
        self._thread.start()

    def _run(self):
        reader = None
        try:
            reader = self._opener(self.path)
            with self._cond:
                self.size = reader.size
                self._cond.notify_all()
            while not self._stop:
                data = reader.read(CHUNK_SIZE)
                if not data:
                    break
                with self._file_lock:
                    self._file.seek(self.done)
                    self._file.write(data)
                    self._file.flush()
                with self._cond:
                    self.done += len(data)
                    self._cond.notify_all()
            if not self._stop and self.done != self.size:
                raise IOError(f'Archive member ended early at {self.done} of {self.size} bytes')
            if not self._stop:
                logger.info(f'Unpacked {os.path.basename(self.path)} ({self.done} bytes)')
        except Exception as e:
            logger.error(f'Unpacking {self.path} failed: {e}')
            with self._cond:
                self.error = e
                self._cond.notify_all()
        finally:
            if reader is not None:
                _close_quietly(reader)

    def read(self, offset, size):
        """Up to `size` bytes at `offset`, waiting until the unpacker has reached them."""
        with self._cond:
            while self.done <= offset and self.error is None and not self._stop:
                self._cond.wait()
            if self.done <= offset:
                raise IOError(f'Cannot read {self.path}: {self.error or "unpacking stopped"}')
            size = min(size, self.done - offset)
        with self._file_lock:
            self._file.seek(offset)
            return self._file.read(size)

    def wait_size(self):
        with self._cond:
            while self.size is None and self.error is None:
                self._cond.wait()
            if self.error is not None and self.size is None:
                raise IOError(f'Cannot read {self.path}: {self.error}')
            return self.size

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=30)
        with self._file_lock:
            self._file.close()


class ExtractionCache:
    """The members being served, one Extraction each, dropped once idle for `ttl` seconds.

    Kept past the end of a download so that the reads an installer comes back for - and the
    retries of reads it gave up on - are answered from the file instead of unpacked again.
    """

    def __init__(self, ttl=600.0, opener=MemberReader):
        self._lock = threading.Lock()
        self._entries = {}
        self._ttl = ttl
        self._opener = opener
        self._reaper = None

    def acquire(self, path):
        stale = []
        with self._lock:
            entry = self._entries.get(path)
            if entry is not None and entry.error is not None:
                stale.append(self._entries.pop(path))
                entry = None
            if entry is None:
                stale += self._make_room_locked()
                entry = self._entries[path] = Extraction(path, self._opener)
            entry.users += 1
            entry.last_used = time.monotonic()
            self._arm_reaper_locked()
        for e in stale:
            e.close()
        return entry

    def release(self, entry):
        with self._lock:
            entry.users -= 1
            entry.last_used = time.monotonic()

    def clear(self):
        with self._lock:
            entries, self._entries = list(self._entries.values()), {}
        for e in entries:
            e.close()

    def _make_room_locked(self):
        """Drop idle extractions, oldest first, while the cap is reached."""
        limit = _cache_limit()
        if not limit:
            return []
        dropped = []
        idle = sorted((e for e in self._entries.values() if e.users == 0), key=lambda e: e.last_used)
        while idle and sum(e.done for e in self._entries.values()) >= limit:
            e = idle.pop(0)
            dropped.append(self._entries.pop(e.path))
        return dropped

    def _expire_locked(self):
        now = time.monotonic()
        stale = [p for p, e in self._entries.items() if e.users == 0 and now - e.last_used >= self._ttl]
        return [self._entries.pop(p) for p in stale]

    def _arm_reaper_locked(self):
        if self._reaper is not None or not self._entries:
            return
        self._reaper = threading.Timer(min(self._ttl, 60.0), self._reap)
        self._reaper.daemon = True
        self._reaper.start()

    def _reap(self):
        with self._lock:
            self._reaper = None
            stale = self._expire_locked()
            self._arm_reaper_locked()
        for e in stale:
            logger.debug(f'Dropping unpacked {e.path}')
            e.close()


def _close_quietly(reader):
    try:
        reader.close()
    except Exception as e:
        logger.debug(f'Error closing archive reader for {getattr(reader, "path", "?")}: {e}')


cache = ExtractionCache()


def stream(path, start, stop):
    """Yield bytes [start, stop) of the member in `path`, out of its shared extraction.

    The extraction is acquired on first iteration, so a response whose body is never sent
    (HEAD) starts no unpacking.
    """
    entry = cache.acquire(path)
    try:
        pos = start
        while pos < stop:
            data = entry.read(pos, min(CHUNK_SIZE, stop - pos))
            pos += len(data)
            yield data
    finally:
        cache.release(entry)
