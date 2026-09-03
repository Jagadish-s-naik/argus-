import os, base64

base_dir = r'c:\Users\Ashwin\OneDrive\Desktop\Argus\argus-\ML Model'

with open(os.path.join(base_dir, 'logo-mark.png'), 'rb') as f:
    b64_mark = base64.b64encode(f.read()).decode('utf-8')

with open(os.path.join(base_dir, 'favicon-32x32.png'), 'rb') as f:
    b64_fav = base64.b64encode(f.read()).decode('utf-8')

dash_path = os.path.join(base_dir, 'dashboard.html')
with open(dash_path, 'r', encoding='utf-8') as f:
    html = f.read()

html = html.replace('src="logo-mark.png"', f'src="data:image/png;base64,{b64_mark}"')
html = html.replace('href="favicon.ico"', f'href="data:image/png;base64,{b64_fav}"')
html = html.replace('href="favicon-32x32.png"', f'href="data:image/png;base64,{b64_fav}"')
html = html.replace('href="favicon-16x16.png"', f'href="data:image/png;base64,{b64_fav}"')
html = html.replace('href="apple-touch-icon.png"', f'href="data:image/png;base64,{b64_fav}"')

with open(dash_path, 'w', encoding='utf-8') as f:
    f.write(html)

print('Embedded Data URIs directly into dashboard.html successfully!')
