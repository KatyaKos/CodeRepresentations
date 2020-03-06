from os import walk
from os.path import isfile, join


if __name__ == '__main__':
    dirname = '/home/katyakos/Briksin_projects/data/train/presto'
    fs = []
    for (dirpath, dirnames, filenames) in walk(dirname):
        fs.extend([join(dirname, f) for f in filenames])
    with open('/home/katyakos/Briksin_projects/data/result/java/short_preprocessed.java', 'w') as fout:
        script = ''
        for filename in fs:
            with open(filename) as fin:
                for line in fin:
                    script += line.strip().lower() + ' EOS '

        things = ['.', '(', ')', '<', '>', '/', '[', ']', '"']
        for t in things:
            words = script.split(t)
            s = ' ' + t + ' '
            script = s.join(words)
        script += ' EOS '
        fout.write(script)
