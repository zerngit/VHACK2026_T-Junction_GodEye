import re

with open("mesa_drone_rescue_langgraph_trace.py", "r", encoding="utf-8") as f:
    content = f.read()

new_style = '''class SidebarStyleElement(base.TextElement):
    def render(self, model):
        return """
        <style>
        /* Modern Light Theme inspired by the user's reference image */
        body {
            background-color: #f0fdfa !important; /* Extremely soft teal/white */
            color: #334155 !important;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;   
        }

        /* Sidebar Container */
        .sidebar, #sidebar-nav, .col-md-3, .col-lg-3 {
            background-color: #ffffff !important;
            border-right: none !important;
            border-radius: 0 16px 16px 0 !important;
            padding: 24px !important;
            box-shadow: 4px 0 24px rgba(20, 184, 166, 0.08) !important;
        }

        /* Override the giant blue Bootstrap labels */
        .label, .badge, .control-label, label {
            background-color: #ccfbf1 !important; /* light cyan/teal background */
            color: #0f766e !important; /* dark teal text */
            font-weight: 700 !important;
            font-size: 11px !important;
            padding: 4px 10px !important;
            border-radius: 12px !important; /* Small rounded pill */
            display: inline-flex !important;
            align-items: center;
            width: fit-content !important; /* Fixes huge full-width bars */
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: none !important;
            margin-top: 10px !important;
            margin-bottom: 6px !important;
            line-height: 1.4 !important;
        }

        /* Checkbox formatting (Simulate AI drives drones) */
        .checkbox {
            margin-top: 24px !important;
            margin-bottom: 30px !important; /* Move LangGraph commander model lower */
        }

        .checkbox label {
            display: flex !important;
            align-items: center !important;
            flex-direction: row-reverse !important;
            justify-content: flex-end !important;
            gap: 12px !important;
            width: 100% !important; 
            background-color: #f8fafc !important; /* Very soft gray background for the row */
            padding: 10px 16px !important;
            border-radius: 16px !important;
            border: 1px solid #e2e8f0 !important;
        }

        /* Selects and inputs */
        select.form-control, select.form-select, input.form-control, select, input[type="text"] {
            background-color: #f8fafc !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 20px !important; /* pill shape */
            color: #0f766e !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            box-shadow: none !important;
            transition: all 0.2s ease !important;
            -webkit-appearance: none !important;
            appearance: none !important;
            width: 100% !important;
            background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2314b8a6%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.4-12.8z%22%2F%3E%3C%2Fsvg%3E") !important;
            background-repeat: no-repeat !important;
            background-position: right 12px top 50% !important;
            background-size: 12px auto !important;
        }

        select.form-control:focus, input.form-control:focus, select:focus {
            border-color: #14b8a6 !important;
            outline: none !important;
            box-shadow: 0 0 0 4px rgba(20, 184, 166, 0.15) !important;
            background-color: #ffffff !important;
        }

        /* Core Buttons */
        .btn {
            border-radius: 20px !important; /* Soft pill shape */
            font-weight: 700 !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
            margin-bottom: 10px !important;
            border: none !important;
        }

        /* Play/Start button */
        .btn-success, #play-pause {
            background-color: #2ed573 !important; 
            color: #ffffff !important;
            box-shadow: 0 4px 10px rgba(46, 213, 115, 0.3) !important;
        }
        .btn-success:hover, #play-pause:hover {
            background-color: #26b360 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 14px rgba(46, 213, 115, 0.4) !important;
        }

        /* Step button */
        .btn-primary, #step {
            background-color: #14b8a6 !important;
            color: #ffffff !important;
            box-shadow: 0 4px 10px rgba(20, 184, 166, 0.3) !important;
        }
        .btn-primary:hover, #step:hover {
            background-color: #0d9488 !important;
            transform: translateY(-2px) !important;
        }

        /* Reset button */
        .btn-default, .btn-secondary, #reset {
            background-color: transparent !important;
            color: #14b8a6 !important;
            border: 2px solid #14b8a6 !important;
        }
        .btn-default:hover, .btn-secondary:hover, #reset:hover {
            background-color: #f0fdfa !important;
            transform: translateY(-2px) !important;
        }

        /* Sliders / Ranges */
        input[type="range"] {
            -webkit-appearance: none !important;
            width: 100% !important;
            height: 8px !important;
            background: #ccfbf1 !important; 
            border-radius: 4px !important;
            outline: none !important;
            margin: 14px 0 !important;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none !important;
            appearance: none !important;
            width: 24px !important;
            height: 24px !important;
            border-radius: 50% !important;
            background: #14b8a6 !important;
            cursor: pointer !important;
            box-shadow: 0 2px 6px rgba(20, 184, 166, 0.4) !important;
            border: 3px solid #ffffff !important;
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2) !important;
        }

        /* Checkboxes -> Toggles (like the image) */
        input[type="checkbox"] {
            appearance: none !important;
            -webkit-appearance: none !important;
            width: 46px !important;
            height: 24px !important;
            background: #cbd5e1 !important; 
            border-radius: 24px !important;
            position: relative !important;
            cursor: pointer !important;
            outline: none !important;
            transition: background 0.3s ease !important;
            vertical-align: middle !important;
            margin: 0 !important;
        }

        input[type="checkbox"]::after {
            content: '' !important;
            position: absolute !important;
            top: 2px !important; /* Adjusted so thumb centers nicely */
            left: 2px !important;
            width: 20px !important;
            height: 20px !important;
            background: #ffffff !important;
            border-radius: 50% !important;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.15) !important;
        }

        input[type="checkbox"]:checked {
            background: #2ed573 !important; 
        }

        input[type="checkbox"]:checked::after {
            transform: translateX(22px) !important;
        }

        /* Map fixes to preserve grid styling */
        .sidebar {
             font-size: 14px;
        }
        </style>
        """
'''

pattern = r'class SidebarStyleElement\(base\.TextElement\):.*?class VideoExportElement\(base\.TextElement\):'
new_content = re.sub(pattern, new_style + '\nclass VideoExportElement(base.TextElement):', content, flags=re.DOTALL)

with open("mesa_drone_rescue_langgraph_trace.py", "w", encoding="utf-8") as f:
    f.write(new_content)
