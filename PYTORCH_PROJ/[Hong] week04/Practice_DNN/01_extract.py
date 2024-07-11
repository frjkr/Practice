from PIL import Image
import struct
import glob
import sys
import os

# http://etlcdb.db.aist.go.jp/specification-of-etl-8

# unzip ETL8G.zip first

JIS_HIRAGANA_FIRST = 0x2421
JIS_HIRAGANA_LAST = 0x2473
JIS_KATAKANA_FIRST = 0x2521
JIS_KATAKANA_LAST = 0x2576

#filename = 'ETL8G_01'
PREFIX = 'ETL8G_PNG'
SZ_RECORD = 8199

def read_record_ETL8G(s):
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)

def is_kana(code):
    if JIS_HIRAGANA_FIRST <= code <= JIS_HIRAGANA_LAST:
        return True
    elif JIS_KATAKANA_FIRST <= code <= JIS_KATAKANA_LAST:
        return True
    else:
        return False

for filename in sorted(glob.glob('ETL8G_??')):
    with open(filename, 'rb') as f:
        while True:
            s = f.read(SZ_RECORD)
            if len(s) != SZ_RECORD:
                break
            r = read_record_ETL8G(s)
    
            #print( r[0:-2], hex(r[1]) )

            sheet = r[0]
            idx = r[3]
            code_jis = r[1]
    

            #if not is_kana(code_jis):
            #    continue
    
            moji = (b'\033$B' + code_jis.to_bytes(2, 'big')).decode('iso2022_jp')
            os.makedirs(os.path.join(PREFIX, 'SRC', moji), exist_ok=True)

            fn = os.path.join(PREFIX, 'SRC', moji, f'{sheet:04d}_{idx:04d}.png')
            print(fn)
    
            iE = Image.eval(r[-1], lambda x: 255-x*16)
            iE.save(fn, 'PNG')
    

