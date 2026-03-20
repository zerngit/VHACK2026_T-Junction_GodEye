import os

old_str = 'short_reason = reason if len(reason) < 120 else reason[:117] + "..."'
new_str = 'short_reason = reason if len(reason) < 120 else reason[:117] + "..."'

for py in os.listdir('.'):
    if not py.endswith('.py'): continue
    with open(py, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if old_str in text:
        text = text.replace(old_str, new_str)
        with open(py, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'Patched {py}')
