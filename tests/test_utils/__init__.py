"""Test-owned helpers that are NOT part of the system under test.

The GeoNames client, response models, service APIs and config loader now live
under src/geonames/ (the SUT owns its own composition via factory.py); tests
import those directly. This package only keeps test infrastructure:
verification engines (ground truth matching, FDSN catalogs, AQuA evaluation)
and transport spies (RecordingSession, user-flow helpers).
"""
