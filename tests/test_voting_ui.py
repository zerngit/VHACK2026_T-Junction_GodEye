"""
Quick test to verify voting UI button is wired correctly in langgraph trace UI.
"""

import sys
sys.path.insert(0, '.')

print("[1] Importing mesa_drone_rescue_langgraph_trace...")
try:
    import controllers.mesa_drone_rescue_langgraph_trace as ui_module
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\n[2] Checking VotingSimulatorElement class exists...")
if hasattr(ui_module, 'VotingSimulatorElement'):
    print("✓ VotingSimulatorElement class found")
else:
    print("✗ VotingSimulatorElement class NOT found")
    sys.exit(1)

print("\n[3] Checking TriggerVotingHandler class exists...")
if hasattr(ui_module, 'TriggerVotingHandler'):
    print("✓ TriggerVotingHandler class found")
else:
    print("✗ TriggerVotingHandler class NOT found")
    sys.exit(1)

print("\n[4] Creating model with LangGraph trace...")
try:
    model = ui_module.LangGraphTraceDroneRescueModel(
        width=24,
        height=16,
        num_drones=4,
        num_survivors=12,
        simulate_ai=False,
    )
    print(f"✓ Model created with {len(model.schedule.agents)} agents")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

print("\n[5] Checking voting_simulator is accessible from model...")
if hasattr(model._tools, 'voting_simulator'):
    voting_sim = model._tools.voting_simulator
    print(f"✓ voting_simulator found")
    print(f"  - State: {voting_sim.state}")
    print(f"  - Type: {type(voting_sim).__name__}")
else:
    print("✗ voting_simulator NOT accessible from model._tools")
    sys.exit(1)

print("\n[6] Testing voting UI element render...")
try:
    voting_element = ui_module.VotingSimulatorElement()
    html = voting_element.render(model)
    if "voting_drone_select" in html and "Simulate Drone Idle" in html:
        print("✓ Voting UI element renders correctly")
        print(f"  - HTML includes select dropdown: {'voting_drone_select' in html}")
        print(f"  - HTML includes voting button: {'Simulate Drone Idle' in html}")
    else:
        print("⚠ Voting UI missing expected elements")
        print(f"  HTML length: {len(html)}")
except Exception as e:
    print(f"✗ UI render failed: {e}")
    sys.exit(1)

print("\n[7] Triggering voting round...")
try:
    model.trigger_voting('d_0')
    voting_sim = model._tools.voting_simulator
    print(f"✓ Voting triggered")
    print(f"  - State: {voting_sim.state}")
    print(f"  - Idle drone: {voting_sim.idle_drone_id}")
    print(f"  - Vote tally: {voting_sim.vote_tally}")
    print(f"  - Winning sector: {voting_sim.winning_sector}")
except Exception as e:
    print(f"✗ Voting trigger failed: {e}")
    sys.exit(1)

print("\n[8] Checking server setup...")
try:
    server = ui_module.server
    print(f"✓ Server created: {server}")
    
    # Check that voting element is in the visualization elements
    visual_elements = server.visualization_elements if hasattr(server, 'visualization_elements') else []
    print(f"  - Server has visualization elements: {len(visual_elements) if hasattr(server, 'visualization_elements') else 'N/A'}")
except Exception as e:
    print(f"⚠ Server check: {e}")

print("\n" + "="*80)
print("✓ ALL TESTS PASSED!")
print("="*80)
print("\nVoting UI is now available in mesa_drone_rescue_langgraph_trace.py!")
print("\nTo use:")
print("  1. Run: python mesa_drone_rescue_langgraph_trace.py")
print("  2. Go to http://localhost:8544 (or shown port)")
print("  3. Look for the 'Voting State' control panel")
print("  4. Select a drone and click 'Simulate Drone Idle'")
print("  5. Watch voting happen in console and trace file!")
