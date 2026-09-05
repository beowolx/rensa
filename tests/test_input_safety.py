import os
import subprocess
import sys

import pytest
from rensa import CMinHash, RMinHash


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Python buffer methods require Python 3.12"
)


class MutatingBuffer:
    def __init__(self, target, phase):
        self.target = target
        self.phase = phase

    def __buffer__(self, flags):
        if self.phase == "acquire":
            self.target.clear()
        return memoryview(b"token")

    def __release_buffer__(self, view):
        if self.phase == "release":
            self.target.clear()


@pytest.mark.parametrize("phase", ["acquire", "release"])
@pytest.mark.parametrize("prefix", [[], ["first"], [b"first"], [bytearray(b"first")]])
@pytest.mark.parametrize("minhash", [RMinHash, CMinHash])
def test_token_list_resize_during_buffer_callback(minhash, prefix, phase):
    tokens = list(prefix)
    tokens.extend([MutatingBuffer(tokens, phase), b"last"])
    with pytest.raises(RuntimeError, match="token list changed size"):
        minhash(num_perm=16, seed=42).update(tokens)


@pytest.mark.parametrize("phase", ["acquire", "release"])
@pytest.mark.parametrize("prefix", [[], [b"first"], [bytearray(b"first")]])
def test_byte_token_list_resize_during_buffer_callback(prefix, phase):
    tokens = list(prefix)
    tokens.extend([MutatingBuffer(tokens, phase), b"last"])
    with pytest.raises(RuntimeError, match="token list changed size"):
        RMinHash.digest_matrix_from_token_byte_sets([tokens], 16, 42)


@pytest.mark.parametrize("phase", ["acquire", "release"])
def test_sampled_token_list_resize_during_buffer_callback(monkeypatch, phase):
    monkeypatch.setenv("RENSA_RHO_TOKEN_BUDGET", "2")
    tokens = []
    tokens.extend(MutatingBuffer(tokens, phase) for _ in range(512))
    with pytest.raises(RuntimeError, match="token list changed size"):
        RMinHash.digest_matrix_from_token_sets_rho([tokens], 16, 42)


@pytest.mark.parametrize("phase", ["acquire", "release"])
def test_rho_fallback_outer_list_resize(phase):
    script = f"""
from rensa import RMinHash
class Exporter:
    def __buffer__(self, flags):
        if {phase!r} == 'acquire':
            corpus.clear()
        return memoryview(b'token')
    def __release_buffer__(self, view):
        if {phase!r} == 'release':
            corpus.clear()
corpus = [[b'first']] + [[Exporter()] for _ in range(31)]
try:
    RMinHash.digest_matrix_from_token_sets_rho(corpus, 16, 42)
except IndexError:
    pass
else:
    raise AssertionError('expected resized outer list to be rejected')
"""
    env = dict(os.environ, RAYON_NUM_THREADS="2", RENSA_DOC_PAR_BATCH_SIZE="32",
               RENSA_RHO_RAW_PARALLEL="1")
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("phase", ["acquire", "release"])
@pytest.mark.parametrize("minhash", [RMinHash, CMinHash])
def test_buffer_destructor_resize_after_callback(minhash, phase):
    tokens = []
    released = []

    class Exporter:
        def __buffer__(self, flags):
            if phase == "acquire":
                tokens[0] = b"replacement"
            return memoryview(b"token")

        def __release_buffer__(self, view):
            if phase == "release":
                tokens[0] = b"replacement"

        def __del__(self):
            released.append(True)
            tokens.clear()

    tokens.extend([Exporter(), b"last"])
    with pytest.raises(RuntimeError, match="token list changed size"):
        minhash(num_perm=16, seed=42).update(tokens)
    assert released == [True]


@pytest.mark.parametrize("minhash", [RMinHash, CMinHash])
def test_same_size_list_replacement_after_subclass_token(minhash):
    tokens = [type("Token", (str,), {})("first")]

    class Exporter:
        def __buffer__(self, flags):
            tokens[:] = [b"replacement", b"replacement", b"last"]
            return memoryview(b"token")

    tokens.extend([Exporter(), "old last"])
    result = minhash(num_perm=16, seed=42)
    result.update(tokens)
    expected = minhash(num_perm=16, seed=42)
    expected.update(["first", b"token", b"last"])
    assert result.digest() == expected.digest()


@pytest.mark.parametrize("queue_cap", ["0", "2"])
@pytest.mark.parametrize("phase", ["acquire", "release"])
def test_classic_matrix_rejects_missing_outer_rows(monkeypatch, queue_cap, phase):
    monkeypatch.setenv("RENSA_PIPELINE_QUEUE_CAP", queue_cap)
    corpus = []
    corpus.extend([[MutatingBuffer(corpus, phase)], [b"last"]])
    with pytest.raises(ValueError, match="document list changed size"):
        RMinHash.digest_matrix_from_token_sets(corpus, 16, 42)
