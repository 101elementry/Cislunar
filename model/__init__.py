"""
model: scenario data structures, JSON serialisation, family files, and
the code that translates a Scenario into engine calls.

This package knows nothing about display.  It imports engine (to run a
scenario and to convert units) and nothing from app.

Modules
  scenario  Scenario, Spacecraft, GroundStation, OpticalSensor, JSON,
            element presets
  family    loading the family files produced by validate.py and
            build_families.py, picking a member by property
  orbits    any Spacecraft to an initial state; correction of a typed
            state to a periodic orbit; orbit description
  runner    run_scenario: propagate every spacecraft, evaluate every
            observer-spacecraft pair, manifolds and coverage
  sweep     run a scenario across a range of one setting
"""
