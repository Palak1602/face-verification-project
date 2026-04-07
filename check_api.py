import requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

eps = ['/capture/live_camera', '/capture/id_camera', '/extract/id_face', '/verify']
for ep in eps:
    try:
        r = requests.post('https://localhost:8000' + ep, data={'camera_index':0}, verify=False, timeout=120)
        print(ep, r.status_code, r.text[:400])
    except Exception as e:
        print(ep, 'ERROR', e)
