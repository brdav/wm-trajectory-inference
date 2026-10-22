#!/usr/bin/env python3
"""HTML spot-check report of an OSM-map image set written by scripts/osm_maps/make_osm_maps.py (stdlib + ImageMagick `convert` only).

One self-contained .html (images embedded as base64 JPEG): the legend (optional), a per-country-folder table (plotted /
unplottable / gate failures / no route / standing / mean speed / share whose map comes from another region) and a gallery
of images: 2 per country folder (US: one per state), plus situation groups (dense urban, sharp low-speed turns, fast roads,
left-hand traffic, standing car). Samples are chosen by a fixed hash of the sample id (a fair sample, not best cases).

Usage:
  python scripts/osm_maps/make_osm_report.py --maps <osm_bev>/maps_predicted --out <dir> [--legend legend.png]
  # the same samples, drawn with the other path, for a side-by-side comparison:
  python scripts/osm_maps/make_osm_report.py --maps <osm_bev>/maps_map_matched --out <dir2> --like <dir>/items.json [--legend legend.png]

--maps   a maps_<path> folder of make_osm_maps.py (needs index.shard*.jsonl / unplottable.shard*.jsonl and the session zips)
--like   items.json of another report: use exactly those samples (needs the same sample ids in --maps)
--legend image written by `osm_bev.py legend` (embedded, downscaled to 1700 px wide)
--label  free text for the heading (e.g. the Slurm job id of the run)
Also writes <out>/items.json (the chosen samples) and <out>/img/*.jpg.
"""
import argparse
import json, glob, os, zipfile, hashlib, heapq, collections, subprocess, base64, html, sys
ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--maps',required=True); ap.add_argument('--out',required=True); ap.add_argument('--like'); ap.add_argument('--legend'); ap.add_argument('--label',default='')
A=ap.parse_args()
B=A.maps.rstrip('/'); OUT=A.out.rstrip('/')
FORCE=json.load(open(A.like)) if A.like else None
FSID={x['sid'] for x in FORCE} if FORCE else set()
FREC={}
KIND='?'
for _f in sorted(glob.glob(B+'/index.shard*.jsonl'))[:1]:
    KIND=json.loads(open(_f).readline())['path']      # predicted | map_matched | none
h=lambda s: hashlib.md5(('report1:'+s).encode()).hexdigest()
def folder(clip):
    p=clip.split('/natix/')[1].split('/'); return (p[0], p[1] if p[0]=='United States' else None)
stats=collections.defaultdict(lambda: collections.Counter()); spd=collections.defaultdict(float)
heaps=collections.defaultdict(list)       # key -> heap of (hash, record) smallest-hash kept, size-limited
K={'country':3,'cat':10}
def keep(key,rec,n):
    x=(h(rec['sample_id']),rec['sample_id']); hp=heaps[key]
    heapq.heappush(hp,(-int(x[0][:12],16),rec['sample_id'],rec))
    if len(hp)>n: heapq.heappop(hp)
tot=0
for f in sorted(glob.glob(B+'/index.shard*.jsonl')):
    for l in open(f):
        r=json.loads(l); tot+=1; c,st=folder(r['clip_id']); s=stats[c]; s['plotted']+=1
        if r['sample_id'] in FSID: FREC[r['sample_id']]=r
        if not r['path_drawn']: s['standing']+=1
        else: spd[c]+=r['mean_speed_kmh']
        reg=r['region'] or ''; s['region_differs']+= (reg.split('_')[0]!=c)
        if r['path_drawn'] and r['mean_speed_kmh']>5: keep(('country',c if c!='United States' else 'US:'+st),r,K['country'])
        if r['path_drawn']:
            v=r['mean_speed_kmh']; t=r['path_turn_deg']
            if v>75: keep(('cat','Fast road (> 75 km/h)'),r,12)
            if 8<v<=40 and t is not None and abs(t)>=70: keep(('cat','Sharp turn at low speed'),r,12)
            if 8<v<=35 and r['n_features']>1800: keep(('cat','Dense urban scene'),r,12)
            if r['drive_side']=='left': keep(('cat','Left-hand traffic'),r,8)
        else: keep(('cat','Standing (no path drawn)'),r,6)
unp=collections.defaultdict(collections.Counter)
for f in sorted(glob.glob(B+'/unplottable.shard*.jsonl')):
    for l in open(f):
        r=json.loads(l); c,_=folder(r['clip_id']); unp[c][r['reason']]+=1
# pick: per country 2 (US: 1 per state, 8 states), categories
picked=collections.OrderedDict()
def take(key,n,uniq=None):
    rs=[x[2] for x in sorted(heaps[key],key=lambda x:x[0],reverse=True)]; return rs[:n]
countries=sorted(c for c in stats)
gal=[]
CATN=['Dense urban scene','Sharp turn at low speed','Fast road (> 75 km/h)','Left-hand traffic','Standing (no path drawn)']
for c in countries:
    if c=='United States':
        us=[(k[1],take(k,1)[0]) for k in heaps if k[0]=='country' and k[1].startswith('US:')]
        us.sort(key=lambda x:h(x[0])); 
        for st,r in us[:8]: gal.append(('United States / '+st[3:],r))
    else:
        for r in take(('country',c),2): gal.append((c,r))
cats=[]
for name in ['Dense urban scene','Sharp turn at low speed','Fast road (> 75 km/h)','Left-hand traffic','Standing (no path drawn)']:
    for r in take(('cat',name),6 if name!='Left-hand traffic' else 4): cats.append((name,r))
if FORCE:
    gal=[(x['g'],FREC[x['sid']]) for x in FORCE if x['g'] not in CATN and x['sid'] in FREC]
    cats=[(x['g'],FREC[x['sid']]) for x in FORCE if x['g'] in CATN and x['sid'] in FREC]
zcache={}
def extract(r,i):
    z,n=r['image'].split('::'); zf=zcache.get(z) or zipfile.ZipFile(os.path.join(B,z)); zcache[z]=zf
    wp=f'{OUT}/img/{i}.webp'; open(wp,'wb').write(zf.read(n)); jp=f'{OUT}/img/{i}.jpg'
    subprocess.check_call(['convert',wp,'-resize','640x640','-quality','88',jp]); os.remove(wp); return jp
os.makedirs(OUT+'/img',exist_ok=True)
items=[]
for i,(g,r) in enumerate(gal+cats): items.append((g,r,extract(r,i)))
def cap(r): return f"{html.escape(r['region'] or '?')} &middot; {r['mean_speed_kmh']:.0f} km/h &middot; turn {r['path_turn_deg'] if r['path_turn_deg'] is not None else 0:+.0f}&deg; &middot; gate err {r['snippet_err_mean_m']:.1f}/{r['snippet_err_max_m']:.1f} m &middot; t0 {r['t0_s']:.0f}s"
def fig(g,r,jp,show_g=True):
    b=base64.b64encode(open(jp,'rb').read()).decode()
    return f'<figure><img loading="lazy" src="data:image/jpeg;base64,{b}"><figcaption><b>{html.escape(g)}</b><br>{cap(r)}<br><span class="id">{html.escape(r["sample_id"].split("#")[0][-45:])} f{r["start_frame"]}</span></figcaption></figure>'
rows=[]
for c in sorted(stats,key=lambda c:-stats[c]['plotted']):
    s=stats[c]; u=sum(unp[c].values()); n=s['plotted']; moving=n-s['standing']
    rows.append(f"<tr><td>{html.escape(c)}</td><td>{n:,}</td><td>{u:,}</td><td>{100*n/(n+u):.0f}%</td><td>{unp[c]['snippet_error']:,}</td><td>{unp[c]['no_usable_route']:,}</td><td>{100*s['standing']/n:.0f}%</td><td>{(spd[c]/moving if moving else 0):.0f}</td><td>{100*s['region_differs']/n:.0f}%</td></tr>")
N=sum(s['plotted'] for s in stats.values()); U=sum(sum(v.values()) for v in unp.values())
LEG=''
if A.legend:
    os.makedirs(OUT,exist_ok=True); subprocess.check_call(['convert',A.legend,'-resize','1700x','-quality','92',OUT+'/legend_embed.jpg'])
    LEG=base64.b64encode(open(OUT+'/legend_embed.jpg','rb').read()).decode()
LEGEND_HTML=('<h2>Legend</h2><details open><summary>Colours, symbols and markings in the map images (enlarged 5&times;)</summary><img class="legend" src="data:image/jpeg;base64,'+LEG+'" alt="legend"></details>') if LEG else ''
page=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OSM map spot-check</title><style>
:root{{--bg:#fff;--fg:#1b1b1b;--mut:#666;--line:#ddd;--acc:#7a00ff}}@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa;--line:#333;--acc:#b98cff}}}}
body{{margin:0 auto;max-width:1180px;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}h1{{font-size:1.5rem}}h2{{margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:.2rem}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:3px 8px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:var(--bg)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}figure{{margin:0}}figure img{{width:100%;height:auto;display:block;border:1px solid var(--line)}}figcaption{{font-size:12px;color:var(--mut)}}.id{{font-size:10px;word-break:break-all}}
.legend{{width:100%;max-width:1100px;height:auto;border:1px solid var(--line);margin-top:8px}}summary{{cursor:pointer;color:var(--mut)}}
.note{{border-left:3px solid var(--acc);padding:.2rem .8rem;color:var(--mut)}}</style></head><body>
<h1>OSM map images &mdash; spot-check report ({KIND.replace('_','-')} path)</h1>
<p>Full run{(' ('+html.escape(A.label)+')') if A.label else ''}: <b>{N:,}</b> images written, <b>{U:,}</b> samples labelled unplottable (<b>{100*N/(N+U):.1f}%</b> plottable of those judged). Every image is 768&times;768 px at 0.5&nbsp;m/px, heading-up, with the <b>{'predicted (fused-pose)' if KIND=='predicted' else 'map-matched (on the matched OSM route)'}</b> ego path of the next 5&nbsp;s in violet (dot per second, arrow at the end). This report shows only maps &mdash; no camera frames &mdash; so it checks that the drawing is plausible, not that it matches the video.</p>
<p class="note">Samples are chosen by a fixed hash (not hand-picked), so the gallery is a fair sample, not the best cases.{' The samples are the SAME ones as in the predicted-path report (report/report.html), so the two can be compared image by image.' if FORCE else ''} Gate error = mean/max distance between the predicted and the map-matched path over the drawn 5&nbsp;s.</p>
{LEGEND_HTML}
<h2>Per country folder</h2><table><tr><th>country folder</th><th>plotted</th><th>unplottable</th><th>pass</th><th>gate fail</th><th>no route</th><th>standing</th><th>mean km/h</th><th>map from other region</th></tr>{"".join(rows)}</table>
<p class="note">"map from other region": the clip's position lies in a different extract than its folder name (mislabelled clips, handled by the GPS lookup).</p>
<h2>Gallery by country (2 per country, US: one per state)</h2><div class="grid">{"".join(fig(g,r,jp) for g,r,jp in items[:len(gal)])}</div>
'''
cur=None; body=''
for g,r,jp in items[len(gal):]:
    if g!=cur: body+=(f'</div><h2>{html.escape(g)}</h2><div class="grid">' if cur else f'<h2>{html.escape(g)}</h2><div class="grid">'); cur=g
    body+=fig(g,r,jp)
page+='<h1 style="margin-top:2.5rem">By situation</h1>'+body+'</div></body></html>'
open(OUT+'/report.html','w').write(page); print('items',len(items),'| html MB %.1f'%(len(page)/1e6),'| countries',len(stats))
json.dump([{'g':g,'img':jp,'region':r['region'],'v':r['mean_speed_kmh'],'turn':r['path_turn_deg'],'sid':r['sample_id']} for g,r,jp in items],open(OUT+'/items.json','w'))
