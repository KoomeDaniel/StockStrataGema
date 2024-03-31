from bs4 import BeautifulSoup
import requests
import pprint

def remove_duplicates(data):
    # Convert list of dictionaries to list of tuples
    data_tuples = [tuple(item.items()) for item in data]

    # Convert list of tuples to set to remove duplicates
    data_set = set(data_tuples)

    # Convert set back to list of dictionaries
    data_no_duplicates = [dict(item) for item in data_set]

    return data_no_duplicates

def bsegainers():
    data = requests.get('https://www.moneycontrol.com/stocks/marketstats/bsegainer/index.php').text
    soup = BeautifulSoup(data, 'lxml')

    allCompanies = soup.find_all('span', class_='gld13 disin')
    companyNames = [row.find('a').text for row in allCompanies]

    tablerow = soup.find_all('tr')
    companyHigh = []
    companyLow = []
    companyClose = []
    companyGain = []
    companyChange = []

    for tr in tablerow:
        high = tr.find_all('td', attrs={'width': 75, 'align': 'right'})
        for i in high:
            companyHigh.append(float(i.text.replace(',', '')))
            break

        low = tr.find_all('td', attrs={'width': 80, 'align': 'right'})
        for i in low:
            companyLow.append(float(i.text.replace(',', '')))
            break

        close = tr.find_all('td', attrs={'width': 85, 'align': 'right'})
        for i in close:
            companyClose.append(float(i.text.replace(',', '')))
            break

        gain = tr.find_all('td', attrs={'width': 45, 'align': 'right', 'class': 'green'})
        for i in gain:
            companyGain.append(float(i.text.replace(',', '')))
            break

        change = tr.find_all('td', attrs={'width': 55, 'align': 'right', 'class': 'green'})
        for i in change:
            companyChange.append(float(i.text.replace(',', '')))
            break

    companyData = []
    for i in range(len(companyNames)):
        companyData.append({
            'company': companyNames[i],
            'high': companyHigh[i],
            'low': companyLow[i],
            'change': companyChange[i],
            'gain_in_per': companyGain[i],
            'close_in_per': companyClose[i]
        })

    new_dict = sorted(companyData, key=lambda i: i['gain_in_per'], reverse=True)
    new_dict = remove_duplicates(new_dict)
    return new_dict[:10]