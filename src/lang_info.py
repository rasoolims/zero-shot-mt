""" Creates a JSON dict associating a language to its metadata.

Note that WikiMatrix uses ISO639-1 language codes, which has been superseded by ISO639-3. This script
converts format 3 to format 1 where possible -- if there are duplicate format 1, we choose the first.
"""

import argparse
import json
from pathlib import Path

from iso639 import Lang, exceptions
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--wals_path', '-w', default='../wals')
parser.add_argument('--out_path', '-o', default='../lang_info.json')

def get_langs_d(lang_path):
    """ Gets the langs dictionary used for other tasks. Other scripts should only need to call this
    function.

    Args:
        lang_path (str): path to [wals-dir]/cldf/languages.csv

    Returns:
        dict: language as the key, genus as the value
    """
    with open(lang_path, 'r') as f:
        langs = json.load(f)
        del langs['__meta__']
        for k in langs.keys():
            langs[k] = f"<{langs[k]['Genus']}>"
    return langs


def get_lang_code(row):
    pt3 = row['ISO639P3code']

    # manually fix inconsistencies between WALS and WikiMatrix
    # TODO: consider getting the language code mappings from https://en.wikipedia.org/wiki/List_of_Wikipedias
    row_id = row['ID']
    if row_id == 'aze':
        return 'az'
    elif row_id == 'gba':
        return 'bar'
    elif row_id == 'ceb':
        return 'ceb'
    elif row_id == 'prs':
        return 'fa'
    elif row_id == 'scr':
        return 'hr'
    elif row_id == 'mal':
        return 'mg'
    elif row_id == 'lge':
        return 'nds'
    elif row_id == 'nor':
        return 'no'
    elif row_id == 'alb':
        return 'sq'
    elif row_id == 'swa':
        return 'sw'
    elif row_id == 'wuc':
        return 'wuu'
    elif row_id == 'mnd':
        return 'zh'
    elif row_id == 'ams':
        return 'ar'
    elif row_id == 'rom':
        print('hihi')
        return 'ro'

    if pd.isna(pt3):
        return ''
    try:
        return Lang(pt3=pt3).pt1
    except exceptions.DeprecatedLanguageValue:
        return ''
    except exceptions.InvalidLanguageValue:
        return ''

# def manual_fix(df):
#     # aze is missing a code in WALS
#     aze_idx = df[df['ID'] == 'aze'].index[0]
#     df.loc[aze_idx, 'ISO639P3code'] = 'aze'

def make_lang_json(wals_path):
    cldf_dir = Path(wals_path + '/cldf')
    df = pd.read_csv(cldf_dir / 'languages.csv')
    # manual_fix(df)

    df['ISO639P1code'] = df.apply(get_lang_code, axis=1)
    df.drop_duplicates('ISO639P1code', inplace=True)
    df = df[df['ISO639P1code'] != ''].set_index('ISO639P1code')
    lang_d = df.to_dict(orient='index')
    for k in lang_d.keys():
        del lang_d[k]['Source'] # don't need this, save some space

    # need to add languages only contained in Wikimatrix
    lang_d['eo'] = dict(Name='Esperanto', Genus="Esperanto")
    lang_d['io'] = dict(Name='Ido', Genus="Esperanto")
    lang_d['la'] = dict(Name='Latin', Genus="Romance", ISO639P3code='lat')
    lang_d['lmo'] = dict(Name='Lombard', Genus="Romance")
    lang_d['mwl'] = dict(Name='Mirandese', Genus="Romance")
    lang_d['nds_nl'] = lang_d['nds']
    lang_d['sh'] = lang_d['hr']
    lang_d['sr'] = lang_d['hr'].copy()
    lang_d['sr']['ISO639P3code'] = 'srp'
    lang_d['simple'] = lang_d['en']
    lang_d['az']['ISO639P3code'] = 'aze'

    lang_d['ja']['Genus'] = 'Japanese'
    lang_d['ko']['Genus'] = 'Korean'

    lang_d['__meta__'] = "v1"
    return lang_d

if __name__ == "__main__":
    args = parser.parse_args()
    lang_d = make_lang_json(args.wals_path)
    with open(args.out_path, 'w') as f:
        json.dump(lang_d, f, indent=2)
    print(f'saved to {args.out_path}')
