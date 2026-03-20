import os

old_str = "style='color: #6c757d; font-style: italic; font-size: 12px; max-width: 600px; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'"
new_str = "style='color: #6c757d; font-style: italic; font-size: 12px; max-width: 600px; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'"

for py in os.listdir('.'):
    if not py.endswith('.py'): continue
    with open(py, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if old_str in text:
        text = text.replace(old_str, new_str)
        with open(py, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'Patched {py}')
