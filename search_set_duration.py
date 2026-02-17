import os
for root, dirs, files in os.walk('video_factory\\.venv\\Lib\\site-packages'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    data = fh.read()
            except:
                continue
            if 'set_duration' in data:
                print(path)

