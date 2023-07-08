from bs4 import BeautifulSoup
import requests

url = 'https://www.newegg.com/gigabyte-geforce-rtx-4060-gv-n4060gaming-oc-8gd/p/N82E16814932628?Description=gpu&cm_re=gpu-_-14-932-628-_-Product'

result = requests.get(url)
doc = BeautifulSoup(result.text, 'html.parser')

prices = doc.find_all(string="$")

# bs4 is made in treelike structure
# so the parent of the $ sign can be found ($ is descendant) and the whole descendant can therefore be found
pp = prices[0].parent

strong = pp.find("strong")
print(strong.string)