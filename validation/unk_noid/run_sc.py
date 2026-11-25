"""Run validation.self_consistency with the Earth ephemeris pinned to astropy 'builtin' (as the v0 paper runs)."""
import sys
import priora.attributable as _att
_att.EPHEMERIS_PREFERENCE = ("builtin",)
assert _att.ephemeris_in_use() == "builtin"
from validation import self_consistency
sys.exit(self_consistency.main(sys.argv[1:]))
