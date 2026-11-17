"""Games kept in RAR archives: indexed by their first volume, served decompressed, never extracted.

The archives here are written by hand in store mode - no tool in CI can create a compressed
one. rarfile reads a stored member by real seeks, so the stream mechanics a compressed member
depends on (forward-only reads, a restart to go back) are covered with a fake reader, and the
real `unrar` pipe by forcing it onto a stored member where the tool is installed.
"""
import os
import types
import zlib

import pytest
import rarfile

from containers import archive
from utils import is_library_file

MEMBER = "Archived Game [0100000000020000][v0].nsp"


def _vint(n):
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _header(body):
    sized = _vint(len(body)) + body
    return zlib.crc32(sized).to_bytes(4, "little") + sized


def write_rar5(path, name, data):
    """A single-volume RAR5 holding `data` stored (-m0) as `name`."""
    name = name.encode()
    main = _header(_vint(1) + _vint(0) + _vint(0))
    file_header = _header(
        _vint(2) + _vint(0x02) + _vint(len(data))       # type, flags: has data, data size
        + _vint(0x04) + _vint(len(data)) + _vint(0o644)  # file flags: has crc, size, attributes
        + zlib.crc32(data).to_bytes(4, "little")
        + _vint(0) + _vint(1)                            # compression: store; host: unix
        + _vint(len(name)) + name
    )
    end = _header(_vint(5) + _vint(0) + _vint(0))
    with open(path, "wb") as f:
        f.write(b"Rar!\x1a\x07\x01\x00" + main + file_header + data + end)


def _unrar_available():
    try:
        rarfile.tool_setup(force=True)
        return True
    except rarfile.RarCannotExec:
        return False


@pytest.fixture(autouse=True)
def empty_pool():
    archive.pool.clear()
    yield
    archive.pool.clear()


# --- Which files the library indexes ---

@pytest.mark.parametrize("name, indexed", [
    ("Game.rar", True),            # old-style set, or a single volume
    ("Game.part1.rar", True),
    ("Game.part01.rar", True),
    ("Game.PART001.RAR", True),
    ("Game.part02.rar", False),    # later volumes belong to the set the first one opens
    ("Game.r00", False),
    ("._Game.rar", False),         # AppleDouble metadata, not an archive
    ("Game.zip", False),
])
def test_only_the_first_volume_is_indexed(name, indexed):
    assert is_library_file(name) is indexed


# --- Reading out of an archive ---

def test_member_names_the_container_inside(tmp_path):
    data = os.urandom(5000)
    path = tmp_path / "set.rar"
    write_rar5(path, f"sub/{MEMBER}", data)
    assert archive.member(str(path)) == (MEMBER, len(data))


def test_archive_without_a_container_is_refused(tmp_path):
    path = tmp_path / "music.rar"
    write_rar5(path, "song.mp3", b"x" * 100)
    with pytest.raises(ValueError):
        archive.member(str(path))


def test_file_info_describes_the_container(tmp_path):
    import titles

    data = os.urandom(5000)
    path = tmp_path / "set.rar"
    write_rar5(path, MEMBER, data)
    info = titles.get_file_info(str(path))
    assert info["filepath"] == str(path)
    assert (info["filename"], info["extension"], info["size"]) == (MEMBER, "nsp", len(data))


def test_unreadable_archive_is_skipped_not_raised(tmp_path):
    import titles

    path = tmp_path / "broken.rar"
    path.write_bytes(b"not an archive")
    assert titles.get_file_info(str(path)) is None


def test_container_opens_through_the_archive(tmp_path):
    """nsz reads the archived container the same as the loose one."""
    from containers.container import open_container, partition_entries

    # A PFS0 with two empty-named entries is enough for the table walk.
    names = b"a.nca\0b.nca\0"
    pfs0 = (b"PFS0" + (2).to_bytes(4, "little") + len(names).to_bytes(4, "little") + bytes(4)
            + (0).to_bytes(8, "little") + (16).to_bytes(8, "little") + (0).to_bytes(4, "little") + bytes(4)
            + (16).to_bytes(8, "little") + (16).to_bytes(8, "little") + (6).to_bytes(4, "little") + bytes(4)
            + names + os.urandom(32))
    loose = tmp_path / MEMBER
    loose.write_bytes(pfs0)
    packed = tmp_path / "set.rar"
    write_rar5(packed, MEMBER, pfs0)

    with open_container(str(loose), meta_only=True) as c:
        expected = partition_entries(c)
    with open_container(str(packed), meta_only=True) as c:
        assert partition_entries(c) == expected


# --- Stream reuse, against a reader that behaves like a compressed member ---

class FakeMember:
    """Forward-only like a decompressor: seeking back counts as a restart."""

    def __init__(self, data, log):
        self.data, self.size, self.pos, self.log = data, len(data), 0, log
        self.closed = False
        log["opens"] += 1

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if offset < self.pos:
            self.log["restarts"] += 1
        self.pos = offset

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else n
        out = self.data[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def close(self):
        self.closed = True


@pytest.fixture
def fake_pool(monkeypatch):
    data = os.urandom(10_000)
    log = {"opens": 0, "restarts": 0}
    pool = archive.StreamPool(opener=lambda path: FakeMember(data, log))
    monkeypatch.setattr(archive, "pool", pool)
    yield types.SimpleNamespace(data=data, log=log, pool=pool)
    pool.clear()


def test_a_follow_up_range_resumes_the_parked_stream(fake_pool):
    assert b"".join(archive.stream("a.rar", 0, 4000)) == fake_pool.data[:4000]
    assert b"".join(archive.stream("a.rar", 4000, 6000)) == fake_pool.data[4000:6000]
    assert b"".join(archive.stream("a.rar", 7000, 8000)) == fake_pool.data[7000:8000]
    assert fake_pool.log == {"opens": 1, "restarts": 0}


def test_a_range_behind_every_stream_starts_a_new_one(fake_pool):
    b"".join(archive.stream("a.rar", 0, 4000))
    assert b"".join(archive.stream("a.rar", 100, 200)) == fake_pool.data[100:200]
    assert fake_pool.log == {"opens": 2, "restarts": 0}


def test_streams_are_not_shared_between_archives(fake_pool):
    b"".join(archive.stream("a.rar", 0, 4000))
    b"".join(archive.stream("b.rar", 4000, 5000))
    assert fake_pool.log["opens"] == 2


def test_an_unsent_body_starts_no_decompressor(fake_pool):
    """HEAD builds the response but never iterates it."""
    archive.stream("a.rar", 0, 4000)
    assert fake_pool.log["opens"] == 0


def test_a_disconnect_parks_the_stream_where_it_got_to(fake_pool, monkeypatch):
    monkeypatch.setattr(archive, "CHUNK_SIZE", 1000)
    body = archive.stream("a.rar", 0, 10_000)
    next(body)
    body.close()
    assert b"".join(archive.stream("a.rar", 1000, 2000)) == fake_pool.data[1000:2000]
    assert fake_pool.log == {"opens": 1, "restarts": 0}


def test_a_stream_at_the_end_is_not_parked(fake_pool):
    b"".join(archive.stream("a.rar", 0, 10_000))
    assert fake_pool.pool._idle == []


def test_idle_streams_expire(fake_pool, monkeypatch):
    b"".join(archive.stream("a.rar", 0, 10))
    reader = fake_pool.pool._idle[0][1]
    clock = archive.time.monotonic() + fake_pool.pool._ttl + 1
    monkeypatch.setattr(archive.time, "monotonic", lambda: clock)
    b"".join(archive.stream("a.rar", 0, 10))
    assert reader.closed


def test_cached_reader_revisits_without_restarting():
    log = {"opens": 0, "restarts": 0}
    data = os.urandom(50_000)
    reader = archive.CachedReader(FakeMember(data, log), block_size=4096)
    reader.seek(30_000)
    assert reader.read(100) == data[30_000:30_100]
    reader.seek(10)
    reader.seek(30_050)
    assert reader.read(100) == data[30_050:30_150]
    reader.seek(0)
    assert reader.read(10) == data[:10]  # never read before: the one restart it may cost
    assert log["restarts"] == 1


# --- Through the shop ---

@pytest.fixture(params=["direct", "unrar"])
def archived(request, shop_app, monkeypatch):
    """A RAR in the fixture library, added the way a scan adds one."""
    import titles
    from db import Files, Libraries, create_file
    from settings import set_shop_settings

    if request.param == "unrar":
        if not _unrar_available():
            pytest.skip("no unrar/unar/7z/bsdtar to pipe through")
        # Through the decompressor pipe, as every compressed member is read.
        monkeypatch.setattr(rarfile, "FORCE_TOOL", True)

    set_shop_settings({"public": True})
    data = os.urandom(3 * 1024 * 1024 + 123)
    folder = os.path.join(shop_app.library_root, "Archived")
    os.makedirs(folder)
    path = os.path.join(folder, "archived-game.part01.rar")
    write_rar5(path, MEMBER, data)
    if request.param == "unrar":
        try:
            reader = archive.MemberReader(path)
            works = reader.read(16) == data[:16]
            reader.close()
        except rarfile.Error:
            works = False
        if not works:
            pytest.skip("the installed tool cannot decode RAR (e.g. 7z without its RAR codec)")
    with shop_app.app.app_context():
        library = Libraries.query.first()
        row = create_file(library.id, path, titles.get_file_info(path))
        row.identified = True
        token = row.download_token
        Files.query.session.commit()
    return types.SimpleNamespace(shop=shop_app, data=data, path=path, token=token)


def test_whole_download(archived):
    resp = archived.shop.client.get(f"/api/download/{archived.token}")
    assert resp.status_code == 200
    assert resp.headers["Accept-Ranges"] == "bytes"
    assert int(resp.headers["Content-Length"]) == len(archived.data)
    assert resp.data == archived.data


@pytest.mark.parametrize("header, start, stop", [
    ("bytes=0-1023", 0, 1024),
    ("bytes=1024-", 1024, None),
    ("bytes=-100", -100, None),
])
def test_range_download(archived, header, start, stop):
    resp = archived.shop.client.get(f"/api/download/{archived.token}", headers={"Range": header})
    data = archived.data
    expected = data[start:stop]
    first = start % len(data)
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == f"bytes {first}-{first + len(expected) - 1}/{len(data)}"
    assert resp.data == expected


def test_range_past_the_end(archived):
    resp = archived.shop.client.get(f"/api/download/{archived.token}",
                                    headers={"Range": f"bytes={len(archived.data) + 10}-"})
    assert resp.status_code == 416


def test_head_reports_the_container_size(archived):
    resp = archived.shop.client.head(f"/api/download/{archived.token}")
    assert resp.status_code == 200
    assert int(resp.headers["Content-Length"]) == len(archived.data)
    assert archive.pool._idle == []


def test_sphaira_lists_and_serves_the_container(archived):
    headers = {"User-Agent": "Sphaira/1.0.6"}
    listing = archived.shop.client.get("/Archived/", headers=headers).get_data(as_text=True)
    assert MEMBER in listing and ".rar" not in listing
    resp = archived.shop.client.get(f"/Archived/{MEMBER}", headers=headers)
    assert resp.status_code == 200
    assert resp.data == archived.data


def test_a_move_keeps_the_container_name(archived):
    from db import Files, update_file_path

    moved = archived.path.replace("archived-game", "renamed")
    os.rename(archived.path, moved)
    with archived.shop.app.app_context():
        update_file_path(None, archived.path, moved)
        row = Files.query.filter_by(filepath=moved).one()
        assert row.filename == MEMBER


def test_delete_removes_every_volume(archived, monkeypatch):
    from db import Files
    from library import delete_library_file

    second = archived.path.replace("part01", "part02")
    with open(second, "wb") as f:
        f.write(b"rest of the set")
    monkeypatch.setattr(archive, "volumes", lambda path: [path, second])
    with archived.shop.app.app_context():
        delete_library_file(Files.query.filter_by(filepath=archived.path).one())
        assert Files.query.filter_by(filepath=archived.path).first() is None
    assert not os.path.exists(archived.path)
    assert not os.path.exists(second)


def test_archives_skip_the_stages_that_rewrite_files(archived):
    import tasks
    from db import Files

    mgmt = {
        "organizer": {"enabled": True},
        "verification": {"enabled": True, "depth": "signature"},
        "compression": {"enabled": True},
    }
    with archived.shop.app.app_context():
        row = Files.query.filter_by(filepath=archived.path).one()
        assert not tasks._needs_organize(row, mgmt)
        assert not tasks._needs_verify(row, mgmt)
        assert not tasks._needs_compress(row, mgmt)
