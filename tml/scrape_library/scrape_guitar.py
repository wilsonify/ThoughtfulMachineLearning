import json
import os
import random
import time

import bs4
import pandas as pd
import requests

baseurl = "https://www.ultimate-guitar.com"
order = "rating_desc"
tab_type = "Chords"


def main():
    try:
        result_df = pd.read_csv("chords-tabs.csv")
    except FileNotFoundError:
        result_df = pd.DataFrame(columns=['tab_url'])
    count = 0

    for page in range(1, 100):

        time.sleep(0.5)
        print(f"page={page}")
        page_url = f"{baseurl}/explore?order={order}&page={page}&type[]={tab_type}"
        print(f"page_url={page_url}")
        resp = requests.get(page_url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs4.BeautifulSoup(resp.text)
        js_store = soup.find_all("div", attrs={'class': ["js-store"]})[0]
        content_dict = json.loads(js_store['data-content'])
        tabs_dict = content_dict['store']['page']['data']['data']['tabs']
        tabs_df = pd.DataFrame(tabs_dict)
        tabs_df['pagination'] = page
        tabs_df['pagination_url'] = page_url
        tab_url_unique = tabs_df['tab_url'].unique()
        tab_url_old_unique = result_df['tab_url'].unique()
        tab_url_to_fetch = set(tab_url_unique).difference(tab_url_old_unique)
        total = len(tab_url_to_fetch)

        for tab_url in tab_url_to_fetch:

            tabs_subset_df = tabs_df[tabs_df['tab_url'] == tab_url]

            print(f"tabs_subset_df.shape={tabs_subset_df.shape}")
            count += 1
            time_to_sleep = random.random() * 5
            print(f"time_to_sleep={time_to_sleep}: {count}/{total} tab_url={tab_url}")
            time.sleep(time_to_sleep)
            tab_resp = requests.get(tab_url, headers={'User-Agent': 'Mozilla/5.0'})
            tab_soup = bs4.BeautifulSoup(tab_resp.text)
            tab = tab_soup.find_all("div", attrs={'class': ["js-store"]})[0]
            tab_dict = json.loads(tab['data-content'])
            tab_data = tab_dict['store']['page']['data']['tab']
            count += 1
            tab_data_df = pd.json_normalize(tab_data)
            tab_data_df = tab_data_df.add_suffix('_tabdata')

            tab_id = tab_data['id']
            song_name = tab_data['song_name']
            artist_name = tab_data['artist_name']

            tab_meta = tab_dict['store']['page']['data']['tab_view']['meta']
            tab_meta_df = pd.json_normalize(tab_meta)
            tab_meta_df = tab_meta_df.add_suffix('_meta')
            tab_meta_df['id'] = tab_id
            tab_data_df['id'] = tab_id

            try:
                capo = tab_meta.get('capo', 0)
            except AttributeError:
                capo = 0

            tab_wiki = tab_dict['store']['page']['data']['tab_view']['wiki_tab']
            tab_wiki_df = pd.json_normalize(tab_wiki)
            tab_wiki_df = tab_wiki_df.add_suffix('_wiki')
            tab_wiki_df['id'] = tab_id

            tabs_subset_df = tabs_subset_df.merge(tab_data_df, on='id', how='left')
            tabs_subset_df = tabs_subset_df.merge(tab_meta_df, on='id', how='left')
            tabs_subset_df = tabs_subset_df.merge(tab_wiki_df, on='id', how='left')

            result_df = pd.concat([result_df, tabs_subset_df], axis=0, ignore_index=True)

            content_str = tab_dict['store']['page']['data']['tab_view']['wiki_tab']['content']
            os.makedirs(f"{artist_name}", exist_ok=True)
            if count > 1:
                break
            with open(f"{artist_name}/{song_name}_capo={capo}.txt", 'w') as tab_file:
                tab_file.write(content_str)

        if count > 1:
            break

    result_df.to_csv("chords-tabs.csv")


if __name__ == "__main__":
    main()
