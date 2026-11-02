"""MTap's side of the MTap↔MBin contract (`mtap/recommender`, phase 1).

The contract lives in MBin: `../MBin/docs/contracts/` (protocol + JSON Schemas). This
package publishes RECOMMENDATION and MTap's CONTRACT_STATUS, answers MBin's
RESYNC_REQUEST, and reads TRADING_PARAMETERS, RECOMMENDATION_OUTCOME and MBin's
CONTRACT_STATUS. The schemas are vendored byte-for-byte in `schemas/contracts/`
(`contract.PINNED` + tests/test_recommender.py keep the copy honest).
"""
