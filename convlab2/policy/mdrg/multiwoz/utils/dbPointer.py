import sqlite3
import os

import numpy as np
import zipfile
from convlab2.util.file_util import cached_path
from convlab2.policy.mdrg.multiwoz.utils.nlp import normalize


def auto_download():
    model_path = os.path.join(os.path.dirname(__file__), os.pardir,  'model')
    data_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
    db_path = os.path.join(os.path.dirname(__file__), os.pardir, 'db')
    root_path = os.path.join(os.path.dirname(__file__), os.pardir)

    urls = {model_path: 'https://convlab.blob.core.windows.net/convlab-2/mdrg_model.zip',
            data_path: 'https://convlab.blob.core.windows.net/convlab-2/mdrg_data.zip',
            db_path: 'https://convlab.blob.core.windows.net/convlab-2/mdrg_db.zip'}

    for path in [model_path, data_path, db_path]:
        if not os.path.exists(path):
            file_url = urls[path]
            print('Downloading from: ', file_url)
            archive_file = cached_path(file_url)
            print('Extracting...')
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_path)

# loading databases
domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']#, 'police']
dbs = {}
auto_download()
for domain in domains:
    db = os.path.join(os.path.dirname(__file__), os.pardir, 'db/{}-dbase.db'.format(domain))
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs[domain] = c


def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0,0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector

def queryResult(domain, turn):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    sql_query = "select * from {}".format(domain)

    flag = True
    #print turn['metadata'][domain]['semi']
    for key, val in turn['metadata'][domain]['semi'].items():
        if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                val2 = val2.replace("'", "''")
                # change query for trains
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                val2 = val2.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    #try:  # "select * from attraction  where name = 'queens college'"
    #print sql_query
    #print domain
    num_entities = len(dbs[domain].execute(sql_query).fetchall())

    return num_entities


def queryResultVenues(domain, turn, real_belief=False):
    # query the db
    sql_query = "select * from {}".format(domain)

    flag = True
    if real_belief == True:
        items = turn.items()
    elif real_belief=='tracking':
        for slot in turn[domain]:
            key = slot[0].split("-")[1]
            val = slot[0].split("-")[2]
            if key == "price range":
                key = "pricerange"
            elif key == "leave at":
                key = "leaveAt"
            elif key == "arrive by":
                key = "arriveBy"
            if val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    val2 = val2.replace("'", "''")
                    if key == 'leaveAt':
                        sql_query += key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    val2 = val2.replace("'", "''")
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

            try:  # "select * from attraction  where name = 'queens college'"
                return dbs[domain].execute(sql_query).fetchall()
            except:
                return []  # TODO test it
        pass
    else:
        items = turn['metadata'][domain]['semi'].items()

    flag = True
    for key, val in items:
        if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                val2 = val2.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " +key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                val2 = val2.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    try:  # "select * from attraction  where name = 'queens college'"
        return dbs[domain].execute(sql_query).fetchall()
    except:
        raise
        return []  # TODO test it


def table_schema(domain):
    return [col[1] for col in dbs[domain].execute("PRAGMA table_info({})".format(domain)).fetchall()]
