from bs4 import BeautifulSoup
import requests

# Using a html file for beautifulsoup
# with open("filename", r) as f:
#     doc = BeautifulSoup(f, "html.parser")

url = "https://www.core77.com/jobs#keywords=industrial%20designer&job_levels=1&job_id=517303"

result = requests.get(url)
doc = BeautifulSoup(result.text, 'html.parser')

tags = doc.find_all(['p', 'div'])
print(tags)
