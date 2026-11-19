from collections import defaultdict
ABBR=['BS', 'ZT', 'JS', 'FXB', 'SPZ', 'FZT', 'SM', 'HL', 'LMJ', 'LH', 'LD', 'ZH', 'JK', 'JT', 'CS', 'FXW', 'XHD']

def per_class_display_map(pred):
    first=defaultdict(dict)
    for f,ds in sorted(pred.items()):
        for d in ds:first[d['class_id']].setdefault(d['id'],int(f))
    return {str(c):{pid:f'{ABBR[c]}{i+1}' for i,pid in enumerate(sorted(ids,key=lambda x:(ids[x],x)))} for c,ids in first.items()}
