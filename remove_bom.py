from pathlib import Path

path = Path('fake_news_detection/app.py')
text = path.read_bytes()
bom = bytes((0xef, 0xbb, 0xbf))
if text.startswith(bom):
    path.write_bytes(text[3:])
    print('BOM removed from app.py')
else:
    print('No BOM found in app.py')
