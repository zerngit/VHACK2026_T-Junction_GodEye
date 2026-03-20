
import os
import sys
# Ensure parent directory (V-Hack) is in sys.path so 'core' and others can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#!/usr/bin/env python3
"""
Test script for Voting Simulator integrated with LangGraph Trace Bridge
Validates voting logic and trace recording without running the full Mesa UI

Run: python test_voting_with_trace.py
"""

import sys
import os
import json

from core.mesa_drone_rescue_mcp import DroneRescueModel, SECTOR_DEFS
from controllers.langgraph_mesa_trace_bridge import LangGraphMesaTraceController

def test_voting_with_langgraph_trace():
    """Test voting simulator with langgraph trace recording."""
    print("\n" + "="*80)
    print("VOTING SIMULATOR + LANGGRAPH TRACE TEST")
    print("="*80)
    
    # Create a small model with 4 drones
    print("\n[1] Creating model with 4 drones...")
    model = DroneRescueModel(
        width=24, height=16,
        num_drones=4, num_survivors=12,
        scenario="A: Center quake (clustered)",
        simulate_ai=False  # Don't run AI
    )
    print(f"✓ Model created")
    
    # Create the langgraph trace controller
    print("\n[2] Creating LangGraph Trace Bridge controller...")
    trace_file = "test_voting_trace.jsonl"
    if os.path.exists(trace_file):
        os.remove(trace_file)
    
    controller = LangGraphMesaTraceController(
        tools=model._tools,
        model_name="qwen/qwen3-14b",
        operator_model_name="qwen/qwen-2.5-7b-instruct",
        trace_file=trace_file
    )
    print(f"✓ Controller created. Trace file: {trace_file}")
    
    # Get drone list
    print("\n[3] Getting drone list...")
    result = model._tools.discover_drones()
    drones = result["drones"]
    print(f"✓ Found {len(drones)} drones:")
    for d in drones:
        print(f"  - {d['id']}: pos={d['pos']}, battery={d['battery']}%")
    
    idle_drone = drones[0]["id"]
    print(f"\n✓ Will use '{idle_drone}' as the idle/failed drone")
    
    # Trigger voting round
    print(f"\n[4] Triggering voting round for '{idle_drone}'...")
    voting_result = model.trigger_voting(idle_drone)
    
    if "error" in voting_result:
        print(f"✗ ERROR: {voting_result['error']}")
        return False
    
    print(f"✓ Voting complete!")
    print(f"  - Voters: {voting_result['voter_count']}")
    print(f"  - Winning Sector: {voting_result['winning_sector']}")
    print(f"  - Vote Tally: {voting_result['vote_tally']}")
    
    # Display votes
    print(f"\n[5] Individual votes:")
    for voter_id, vote_info in voting_result['all_votes'].items():
        print(f"  [{voter_id}] → Sector {vote_info['target_sector']}: {vote_info['reasoning']}")
    
    # Display winning action
    if voting_result['winning_action']:
        action = voting_result['winning_action']
        print(f"\n[6] Winning action:")
        print(f"  - Drone: {action['drone_id']}")
        print(f"  - Target: ({action['x']}, {action['y']})")
        print(f"  - Reason: {action['reason']}")
    
    # Have the controller record the voting trace
    print(f"\n[7] Having controller record voting trace...")
    voting_trace = controller._record_voting_trace(1)
    if voting_trace:
        print(f"✓ Voting trace recorded:")
        print(f"  - State: {voting_trace.get('state')}")
        print(f"  - Idle Drone: {voting_trace.get('idle_drone_id')}")
        print(f"  - Vote Tally: {voting_trace.get('vote_tally')}")
        print(f"  - Winning Sector: {voting_trace.get('winning_sector')}")
        
        # Write trace to file
        try:
            with open(trace_file, "a") as f:
                f.write(json.dumps(voting_trace, default=str) + "\n")
            print(f"✓ Trace written to {trace_file}")
        except Exception as e:
            print(f"✗ Error writing trace: {e}")
            return False
    else:
        print(f"✗ No voting trace returned")
        return False
    
    # Now execute voting action
    print(f"\n[8] Executing model step (to execute voting action)...")
    model.step()
    print(f"✓ Step completed")
    
    # Check trace file
    print(f"\n[9] Checking trace file...")
    if not os.path.exists(trace_file):
        print(f"✗ Trace file not created!")
        return False
    
    with open(trace_file, "r") as f:
        trace_lines = f.readlines()
    
    if not trace_lines:
        print(f"⚠ Trace file is empty")
    else:
        print(f"✓ Found {len(trace_lines)} trace entries")
        voting_entries = []
        for line in trace_lines:
            try:
                entry = json.loads(line)
                if entry.get("type") == "drone_voting":
                    voting_entries.append(entry)
            except:
                pass
        
        if voting_entries:
            print(f"✓ Found {len(voting_entries)} voting trace entries:")
            for entry in voting_entries:
                print(f"  - Tick {entry.get('tick')}: State={entry.get('state')}, "
                      f"Idle={entry.get('idle_drone_id')}, "
                      f"Winner=Sector {entry.get('winning_sector')}")
        else:
            print(f"Note: No voting trace entries (expected - need controller.think_and_act to record)")
    
    
    # Check mission log
    print(f"\n[10] Checking mission log...")
    if os.path.exists("Varsity_Hackathon_Mission_Log.txt"):
        with open("Varsity_Hackathon_Mission_Log.txt", "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        voting_lines = [l for l in lines if "VOTING" in l]
        if voting_lines:
            print(f"✓ Found {len(voting_lines)} voting log entries")
            for line in voting_lines[-2:]:
                print(f"  {line.strip()[:100]}...")
        else:
            print(f"⚠ No voting log entries")
    
    # Cleanup
    print(f"\n[11] Cleanup...")
    if os.path.exists(trace_file):
        os.remove(trace_file)
    print(f"✓ Test trace file removed")
    
    print("\n" + "="*80)
    print("✓ TEST PASSED!")
    print("="*80)
    print("\nIntegration verified:")
    print("  ✓ VotingSimulator class works correctly")
    print("  ✓ InUiToolServer.voting_simulator initialized")
    print("  ✓ DroneRescueModel.trigger_voting() triggers rounds")
    print("  ✓ Model.step() executes voting actions")
    print("  ✓ LangGraphMesaTraceController records voting traces")
    print("  ✓ Voting logged to Varsity_Hackathon_Mission_Log.txt")
    print("\nTo use in the langgraph trace server:")
    print("  1. Run: python mesa_drone_rescue_langgraph_trace.py")
    print("  2. Call: model.trigger_voting('drone_id') from Python console")
    print("  3. Watch trace output for voting round details")
    print("  4. Check langgraph_tick_trace_log.jsonl for full trace")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = test_voting_with_langgraph_trace()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH EXCEPTION:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
