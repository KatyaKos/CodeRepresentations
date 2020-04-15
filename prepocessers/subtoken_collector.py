from os import walk
from os.path import isfile, join
import json

def collect(dirname):
    fs = []
    for (dirpath, dirnames, filenames) in walk(dirname):
        fs.extend([join(dirpath, f) for f in filenames])
    
    dic = {}
    for filename in fs:
        extend_dictionary(dic, filename)
    dic['null'] = dic.get('null', 0) + 100
    dic = [(k, dic[k]) for k in dic]
    extras = {}
    for i, p1 in enumerate(dic):
        el1, _ = p1
        l_el1 = len(el1)
        for j in range(i + 1, len(dic)):
            el2, count2 = dic[j]
            pos = el2.find(el1)
            if pos != -1:
                p1, p2 = el2[:pos], el2[pos + l_el1:]
                if len(p1) > 2:
                    extras[p1] = extras.get(p1, 0) + count2
                if len(p2) > 2:
                    extras[p2] = extras.get(p2, 0) + count2
                extras[el1] = extras.get(el1, 0) + count2
    dic = {arr[0]: arr[1] for arr in dic}
    for k in extras:
        dic[k] = dic.get(k, 0) + extras[k]
    dic = [(k, dic[k]) for k in dic]
    dic.sort(key = lambda p: -p[1])
    dic = dic[:15000]
    return [p[0] for p in dic]


def extend_dictionary(dic, filename):
    with open(filename) as fin:
        fin.readline()
        for line in fin:
            token = line.strip().split(',')[-2]
            if token == 'null' or token == '':
                continue
            subtokens = token.split('|')
            for sub in subtokens:
                if sub == '' or len(sub) < 3 or len(sub) > 15:
                    continue
                dic[sub] = dic.get(sub, 0) + 1
    


def write_dict(dic, filename):
    with open(filename, 'w') as fout:
        for tok in dic:
            fout.write(tok + '\n')


if __name__ == "__main__":
    dirname = '/home/ec2-user/subtoken_embedding/data/java-small/result/java/description/'
    dic = collect(dirname)
    write_dict(dic, '/home/ec2-user/subtoken_embedding/data/java-small/result/java/subtokens_java_small_15k.txt')
