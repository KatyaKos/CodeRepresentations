from os import walk
from os.path import isfile, join


if __name__ == '__main__':
    dirname = '/home/ec2-user/subtoken_embedding/data/java-small/training/'
    fs = []
    oi = 0
    for (dirpath, dirnames, filenames) in walk(dirname):
        fs.extend([join(dirpath, f) for f in filenames])
    with open('/home/ec2-user/subtoken_embedding/data/java-small/result/java/preprocessed_java_small.java', 'w') as fout:
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
