import requests
import matplotlib.pyplot as plt

content_type = 'application/json'
headers = {'content-type': content_type}

url = "http://localhost:50001/prediction"


path_object1 = "/Users/zhiming05/test/garrett_cat.jpg"
path_object2 = "/Users/zhiming05/test/burger.jpg"
path_object3 = "/Users/zhiming05/test/sf_bridge.jpg"


img = plt.imread(path_object2)
plt.imshow(img)

img_arr = img.tolist()

data = {"img_arr": img_arr}
response = requests.post(url, json=data, headers=headers)
print(response.json())