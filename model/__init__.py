"""
model: scenario data structures, JSON serialisation, and the runner that
translates a Scenario into engine calls.

This package knows nothing about display.  It imports engine (to run a
scenario and to convert units) and nothing from app.

Modules
  scenario  Scenario, Spacecraft, GroundStation, OpticalSensor, JSON
  family    loading the halo family file produced by validate.py
  runner    run_scenario: propagate every spacecraft, evaluate every
            observer-spacecraft pair through the engine
"""
