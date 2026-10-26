#!/usr/bin/env python3
"""
Label-free mp4 timeline audit. Reports:
- elst (edit list) entries per track (media_time offset, rate, duration)
- stsz sample count per track (true frame / audio sample count)
- mdhd timescale + duration per track
Containers start_time = 0 in both files, but elst can shift a track's
presentation relative to the container timeline.
"""
import struct, sys, os

def boxes(f, start, end):
    pos = start
    while pos < end:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8: break
        size = struct.unpack('>I', hdr[:4])[0]
        typ = hdr[4:8].decode('latin1','replace')
        hsz = 8
        if size == 1:
            size = struct.unpack('>Q', f.read(8))[0]; hsz = 16
        elif size == 0:
            f.seek(0,2); size = f.tell() - pos
        if size < hsz: break
        yield typ, pos, size, hsz
        pos += size

def walk(f, start, end, depth=0, out=None):
    for typ, pos, size, hsz in boxes(f, start, end):
        out.append((depth, typ, pos, size))
        if typ in ('moov','trak','mdia','minf','stbl','edts','dinf'):
            walk(f, pos+hsz, pos+size, depth+1, out)

for path in sys.argv[1:]:
    print(f'=== {os.path.basename(path)} ===')
    with open(path,'rb') as f:
        f.seek(0,2); fsize=f.tell()
        tree=[]
        walk(f,0,fsize,0,tree)
        # find trak boundaries
        traks=[]
        cur=None
        for depth,typ,pos,size in tree:
            if typ=='trak' and depth==1:
                cur={'start':pos,'size':size,'children':[]}
                traks.append(cur)
            if cur is not None and depth>=2 and pos<cur['start']+cur['size']:
                cur['children'].append((typ,pos,size))
        for i,tr in enumerate(traks):
            print(f'-- trak #{i} @{tr["start"]} size={tr["size"]}')
            for typ,pos,size in tr['children']:
                if typ=='hdlr':
                    f.seek(pos+8+4+4); h=f.read(4); print(f'   hdlr={h!r}')
                if typ=='mdhd':
                    f.seek(pos+8); ver=f.read(1)[0]; f.read(3)
                    if ver==1:
                        f.read(16); ts=struct.unpack('>I',f.read(4))[0]; dur=struct.unpack('>Q',f.read(8))[0]
                    else:
                        f.read(8); ts=struct.unpack('>I',f.read(4))[0]; dur=struct.unpack('>I',f.read(4))[0]
                    print(f'   mdhd timescale={ts} duration={dur} -> {dur/ts:.4f}s')
                if typ=='elst':
                    f.seek(pos+8); ver=f.read(1)[0]; f.read(3)
                    n=struct.unpack('>I', f.read(4))[0]
                    print(f'   elst entries={n}')
                    for k in range(n):
                        if ver==1:
                            seg=struct.unpack('>Q', f.read(8))[0]
                            mt=struct.unpack('>q', f.read(8))[0]
                            rate=struct.unpack('>I', f.read(4))[0]
                        else:
                            seg=struct.unpack('>I', f.read(4))[0]
                            mt=struct.unpack('>i', f.read(4))[0]
                            rate=struct.unpack('>I', f.read(4))[0]
                        rhi = rate >> 16
                        rlo = rate & 0xffff
                        print(f'     [{k}] segment_duration={seg} media_time={mt} rate={rhi}.{rlo:04d}')
                if typ=='stsz':
                    f.seek(pos+8+4); f.read(4)  # version/flags
                    sample_size=struct.unpack('>I', f.read(4))[0]
                    count=struct.unpack('>I', f.read(4))[0]
                    print(f'   stsz uniform_size={sample_size} n_samples={count}')
