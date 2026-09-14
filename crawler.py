import requests
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO

url="https://api.inaturalist.org/v1/observations"
E_plant=["Aloe","Zingiber","Mentha","Perilla","Turmeric"]
plant=[126882,122971,54570,119136,119386]
e=4

name = plant[e]
page = 1
number=0

os.chdir(os.path.join(os.path.dirname(__file__), 'picture'))
os.makedirs(E_plant[e], exist_ok=True)
print("資料夾建立成功！")

while True:
    params = {
        "taxon_id": name,
        "photos": "true",
        "per_page": 200,
        "page": page
    }

    data=requests.get(url,params=params).json()
    results=data["results"]

    if len(results)==0 or number==2000:
        break

    img_urls=[]

    for obs in data['results']:
        imgs = obs["photos"][0]["url"].replace(
            "square",
            "original"
        )
        img_urls.append([imgs,number])
        number+=1

    def download(url):
        img_file = requests.get(url[0])
        img=Image.open(BytesIO(img_file.content))
        img.convert('RGB').save(f'{E_plant[e]}/{url[1]}.png')
        img.close()

    executor = ThreadPoolExecutor()
    with ThreadPoolExecutor() as executor:
        executor.map(download, img_urls)

    page+=1

print(f"{E_plant[e]}下載完畢")


