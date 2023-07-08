from bs4 import BeautifulSoup
import requests

# Using a html file for beautifulsoup
# with open("filename", r) as f:
#     doc = BeautifulSoup(f, "html.parser")

url = "https://coinmarketcap.com/"

result = requests.get(url).text
doc = BeautifulSoup(result, 'html.parser')

tbody = doc.tbody
trs = tbody.contents

# print(trs[1].previousSibling)
# print(list(trs[0].descendants))

prices = {}

for tr in trs[:10]:
    name, price = tr.contents[2:4]
    fixed_name = name.p.string
    fixed_price = float(price.a.string.replace('$', '').replace(',', ''))

    prices[fixed_name] = fixed_price

print(prices)