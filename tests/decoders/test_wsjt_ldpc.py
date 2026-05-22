"""LDPC(174,91) decoder regression tests against a reference vector set.

Each vector pins the full 174-LLR input and the 174-bit hard decision the
reference BP decoder produces, so any divergence in the Padé approximations
or BP message scheduling will trip these tests immediately.
"""

import numpy as np
import pytest

from tsdr.radio.decoders.wsjt.ldpc import bp_decode, ldpc_check, ldpc_decode
from tsdr.radio.decoders.wsjt.tables import LDPC_N

from . import _wsjt_ldpc_vectors as v


def _bits_to_array(s: str) -> np.ndarray:
    return np.fromiter((int(c) for c in s), dtype=np.uint8, count=LDPC_N)


@pytest.mark.parametrize(
    "llr_name, expected_name, errors_name",
    [
        ("LLR_clean_arbitrary", "EXPECTED_clean_arbitrary", "ERRORS_clean_arbitrary"),
        ("LLR_clean_zero_payload", "EXPECTED_clean_zero_payload", "ERRORS_clean_zero_payload"),
        ("LLR_noisy_6flips", "EXPECTED_noisy_6flips", "ERRORS_noisy_6flips"),
        ("LLR_noisy_12flips_weak", "EXPECTED_noisy_12flips_weak", "ERRORS_noisy_12flips_weak"),
    ],
)
def test_bp_decode_matches_reference(llr_name: str, expected_name: str, errors_name: str) -> None:
    llr = np.array(getattr(v, llr_name), dtype=np.float32)
    expected = _bits_to_array(getattr(v, expected_name))
    expected_errors = int(getattr(v, errors_name))

    plain, errors = bp_decode(llr, max_iters=25)

    assert plain.shape == (LDPC_N,)
    np.testing.assert_array_equal(plain, expected)
    assert errors == expected_errors


def test_ldpc_check_on_decoded_codeword_returns_zero() -> None:
    """Decoded clean codeword passes parity."""
    llr = np.array(v.LLR_clean_arbitrary, dtype=np.float32)
    plain, errors = bp_decode(llr, max_iters=25)
    assert errors == 0
    assert ldpc_check(plain) == 0


def test_ldpc_check_flags_bit_flip() -> None:
    """Flipping any single bit creates parity violations."""
    llr = np.array(v.LLR_clean_arbitrary, dtype=np.float32)
    plain, _ = bp_decode(llr, max_iters=25)
    plain[0] ^= 1
    # bit 0 appears in 3 parity checks (every variable does), so we expect ~3 errors
    assert ldpc_check(plain) > 0


def test_ldpc_decode_sumproduct_matches_clean_case() -> None:
    """Alternate sum-product decoder agrees with BP on the clean noise-free case."""
    llr = np.array(v.LLR_clean_arbitrary, dtype=np.float32)
    bp_plain, bp_err = bp_decode(llr, max_iters=25)
    sp_plain, sp_err = ldpc_decode(llr, max_iters=25)
    np.testing.assert_array_equal(sp_plain, bp_plain)
    assert sp_err == bp_err


def test_bp_decode_rejects_wrong_size() -> None:
    with pytest.raises(ValueError):
        bp_decode(np.zeros(170, dtype=np.float32))
