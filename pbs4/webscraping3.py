import requests
from bs4 import BeautifulSoup
import re

search_term = input("What GPU?: ")
search_term = search_term.replace(' ', '_')

url = f'https://www.newegg.com/p/pl?d={search_term}&N=4131'
result = requests.get(url).text
doc = BeautifulSoup(result, 'html.parser')

page_text = doc.find(class_='list-tool-pagination-text').strong
pages = int(str(page_text).split('/')[-2].split('>')[-1][:-1])

items_found = {}

for page in range(1, pages+1):
    url = f'https://www.newegg.com/p/pl?d={search_term}&N=4131&page={page}'
    page = requests.get(url).text
    doc = BeautifulSoup(page, 'html.parser')
    div = doc.find(class_='item-cells-wrap border-cells items-grid-view four-cells expulsion-one-cell')

    items = div.find_all(string=re.compile(search_term))
    for item in items:
        parent = item.parent
        link = None
        if parent.name != 'a':
            continue
        link = parent['href']
        next_parent = item.find_parent(class_='item-container')

        if next_parent.find(class_='price-current').strong == None:
            # Prevents breaking due to nonetypes in the grid (items without prices/ads)
            break
        price1 = next_parent.find(class_='price-current').strong.string.replace(',', '')
        price2 = next_parent.find(class_='price-current').sup.string
        price = float(price1 + price2)
        
        items_found[item] = {'price': price, 'link': link}
    
sres = sorted(items_found.items(), key=lambda x: x[1]['price'])
for item in sres:
    print(item[0])
    print(f"${item[1]['price']}")
    print(item[1]['link'])
    print('----------------------------------------------------------------')
