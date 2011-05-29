from operator import itemgetter
from tagassess.dao.annotations import AnnotReader
from tagassess.index_creator import *

def sort_by_len(index):
    bala = []
    for user, items in index.items():
        bala.append( (len(items), user) )
    return sorted(bala, reverse=True)

with AnnotReader('/tmp/test.h5') as r:
    i = r.iterate('delicious')
    idx = create_occurrence_index(i, 'user', 'item')

for siz, user in sort_by_len(idx):
    print user, siz
