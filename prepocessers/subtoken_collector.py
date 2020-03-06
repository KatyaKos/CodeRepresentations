import json

def collect(filename):
    with open(filename) as fin:
        dic = set()
        fin.readline()
        for line in fin:
            token = line.strip().split(',')[-2]
            if token == 'null':
                continue
            subtokens = token.split('|')
            for sub in subtokens:
                dic.add(sub)
        dic.add('null')
        return list(dic)


def write_dict(dic, filename):
    with open(filename, 'w') as fout:
        for tok in dic:
            fout.write(tok + '\n')


if __name__ == "__main__":
    filename = '/home/katyakos/Briksin_projects/data/result/java/short_description.csv'
    dic = collect(filename)
    write_dict(dic, '/home/katyakos/Briksin_projects/data/result/java/short_subtokens.txt')
