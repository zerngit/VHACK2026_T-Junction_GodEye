import re

with open("mesa_drone_rescue_langgraph_trace.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace button styles
content = re.sub(
    r'(?s)style="padding:6px\s+14px;background:#0078d4;color:white;border:none;\s*border-radius:4px;cursor:pointer;font-weight:bold;"',
    r'style="padding:6px 14px;background:#0078d4;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;"',
    content
)
content = re.sub(
    r'(?s)style="padding:6px\s+14px;background:#cc4400;color:white;border:none;\s*border-radius:4px;cursor:pointer;"',
    r'style="padding:6px 14px;background:#cc4400;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;"',
    content
)
content = re.sub(
    r'(?s)style="padding:6px\s+14px;background:#6b21a8;color:white;border:none;\s*border-radius:4px;cursor:pointer;"',
    r'style="padding:6px 14px;background:#6b21a8;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;"',
    content
)
content = re.sub(
    r'(?s)style="padding:6px\s+14px;background:#dc2626;color:white;border:none;\s*border-radius:4px;cursor:pointer;font-weight:bold;"',
    r'style="padding:6px 14px;background:#dc2626;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;"',
    content
)
content = re.sub(
    r'(?s)style="padding:6px\s+14px;background:#15803d;color:white;border:none;\s*border-radius:4px;cursor:pointer;"',
    r'style="padding:6px 14px;background:#15803d;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;"',
    content
)

# Also fix the voting flow button styles if needed
content = re.sub(
    r'(?s)style="padding:6px\s+14px;background:#e94560;color:white;border:none;\s*border-radius:4px;cursor:pointer;font-weight:bold;"',
    r'style="padding:6px 14px;background:#e94560;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;"',
    content
)
content = re.sub(
    r'(?s)style="padding:6px\s+14px;background:#0ea5e9;color:white;border:none;\s*border-radius:4px;cursor:pointer;font-weight:bold;"',
    r'style="padding:6px 14px;background:#0ea5e9;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;"',
    content
)
content = re.sub(
    r'(?s)style="padding:6px\s+14px;background:#2563eb;color:white;border:none;\s*border-radius:4px;cursor:pointer;font-weight:bold;"',
    r'style="padding:6px 14px;background:#2563eb;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;"',
    content
)
content = re.sub(
    r'(?s)style="padding:6px\s+14px;background:#334155;color:white;border:none;\s*border-radius:4px;cursor:pointer;"',
    r'style="padding:6px 14px;background:#334155;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;"',
    content
)

# Second task: Move map and chart above voting simulator in list
# Original array: [video_export, voting_simulator, legend, movement_board, grid, chart]
# Desired array: [video_export, legend, movement_board, grid, chart, voting_simulator]
old_arr = "[video_export, voting_simulator, legend, movement_board, grid, chart]"
new_arr = "[video_export, legend, movement_board, grid, chart, voting_simulator]"
content = content.replace(old_arr, new_arr)

with open("mesa_drone_rescue_langgraph_trace.py", "w", encoding="utf-8") as f:
    f.write(content)


with open("mesa_drone_rescue_mcp.py", "r", encoding="utf-8") as f:
    mcp_content = f.read()

# Third task: structure MCP drone command legend into side-by-side cells
old_legend = '''        return f"""
        <div style="font-family:Arial;line-height:1.6;padding-left:10px;">
          <h3>🤖 MCP Drone Command</h3>
          <div style="padding:8px;background:#e8f4fd;border-radius:5px;
                      border-left:4px solid #0078d4;margin-bottom:8px;">
            <strong>Fleet</strong><br/>{drone_rows}
          </div>
          <div style="padding:8px;background:#f0f0f0;border-radius:5px;
                      margin-bottom:8px;">
            <strong>Mission</strong><br/>
            Sectors: {sectors_done}/6<br/>
            Survivors: {found}/{total_s}
          </div>
          <div style="padding:8px;background:#fff3e0;border-radius:5px;
                      border-left:4px solid #ff6600;margin-bottom:8px;">
            <strong>🏙️ City Buildings</strong><br/>
            <span style="color:#FF69B4;font-weight:bold;">■</span> High (obstacle): {high_count}<br/>
            <span style="color:#87CEEB;font-weight:bold;">■</span> Low (flyable): {low_count}
          </div>
          <h4>Sector Colours</h4>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                      gap:2px;font-size:12px;">
            <div style="background:#e8f0fe;padding:2px 4px;">NW</div>
            <div style="background:#e6f4ea;padding:2px 4px;">N</div>
            <div style="background:#fef7e0;padding:2px 4px;">NE</div>
            <div style="background:#fce8e6;padding:2px 4px;">SW</div>
            <div style="background:#f3e8fd;padding:2px 4px;">S</div>
            <div style="background:#fff3e0;padding:2px 4px;">SE</div>
          </div>
          <div style="margin-top:8px;font-size:13px;">
            ⚡ Charging station<br/>
            🆘 Survivor (detected)<br/>
            ⚪ Survivor (undetected)<br/>
            🚁 Drone (colour = battery)<br/>
            <span style="color:#FF69B4;">🏙️</span> High building (PINK = blocked)<br/>
            <span style="color:#87CEEB;">🏠</span> Low building (SKY BLUE = flyable)
          </div>
        </div>
        """'''

new_legend = '''        return f"""
        <div style="font-family:Arial;line-height:1.6;padding-left:10px;">
          <h3>🤖 MCP Drone Command</h3>
          <div style="display:flex;gap:10px;">
            <div style="flex:1;">
                <div style="padding:8px;background:#e8f4fd;border-radius:5px;
                            border-left:4px solid #0078d4;margin-bottom:8px;">
                  <strong>Fleet</strong><br/>{drone_rows}
                </div>
                <div style="padding:8px;background:#f0f0f0;border-radius:5px;
                            margin-bottom:8px;">
                  <strong>Mission</strong><br/>
                  Sectors: {sectors_done}/6<br/>
                  Survivors: {found}/{total_s}
                </div>
                <div style="padding:8px;background:#fff3e0;border-radius:5px;
                            border-left:4px solid #ff6600;margin-bottom:8px;">
                  <strong>🏙️ City Buildings</strong><br/>
                  <span style="color:#FF69B4;font-weight:bold;">■</span> High (obstacle): {high_count}<br/>
                  <span style="color:#87CEEB;font-weight:bold;">■</span> Low (flyable): {low_count}
                </div>
            </div>
            <div style="flex:1;padding:8px;background:#fafafa;border-radius:5px;
                        border-left:4px solid #ccc;margin-bottom:8px;">
                <h4 style="margin-top:0;">Sector Colours</h4>
                <div style="display:grid;grid-template-columns:1fr 1fr;
                            gap:4px;font-size:12px;text-align:center;">
                  <div style="background:#e8f0fe;padding:2px 4px;font-weight:bold;">NW</div>
                  <div style="background:#e6f4ea;padding:2px 4px;font-weight:bold;">N</div>
                  <div style="background:#fef7e0;padding:2px 4px;font-weight:bold;">NE</div>
                  <div style="background:#fce8e6;padding:2px 4px;font-weight:bold;">SW</div>
                  <div style="background:#f3e8fd;padding:2px 4px;font-weight:bold;">S</div>
                  <div style="background:#fff3e0;padding:2px 4px;font-weight:bold;">SE</div>
                </div>
                <div style="margin-top:12px;font-size:13px;line-height:1.8;">
                  ⚡ Charging station<br/>
                  🆘 Survivor (detected)<br/>
                  ⚪ Survivor (undetected)<br/>
                  🚁 Drone (colour = battery)<br/>
                  <span style="color:#FF69B4;">🏙️</span> High building (PINK = blocked)<br/>
                  <span style="color:#87CEEB;">🏠</span> Low building (SKY BLUE = flyable)
                </div>
            </div>
          </div>
        </div>
        """'''

mcp_content = mcp_content.replace(old_legend, new_legend)

with open("mesa_drone_rescue_mcp.py", "w", encoding="utf-8") as f:
    f.write(mcp_content)

print("Done")
