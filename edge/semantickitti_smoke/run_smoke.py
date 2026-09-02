#!/usr/bin/env python3
import argparse,json,statistics,sys,time
from pathlib import Path
import torch, torch.nn as nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime'))
from pointcept.models.builder import build_model
import pointcept.models.point_transformer_v3.point_transformer_v3m3_utonia
from pointcept.models.utils.structure import Point
class EdgeSegmentor(nn.Module):
    def __init__(self,b,o,n):
        super().__init__(); self.backbone=build_model(b); self.seg_head=nn.Linear(o,n)
    def forward(self,d):
        p=Point(d); p=self.backbone(p)
        if isinstance(p,Point):
            while 'pooling_parent' in p.keys():
                parent=p.pop('pooling_parent'); inverse=p.pop('pooling_inverse')
                parent.feat=torch.cat([parent.feat,p.feat[inverse]],dim=-1); p=parent
            f=p.feat
        else: f=p
        return self.seg_head(f)
def pct(v,p):
    v=sorted(v)
    if len(v)==1:return v[0]
    r=(len(v)-1)*p; lo=int(r); hi=min(lo+1,len(v)-1); q=r-lo
    return v[lo]*(1-q)+v[hi]*q
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variant',choices=['baseline','targetmocha'],default='baseline'); ap.add_argument('--warmup',type=int,default=3); ap.add_argument('--runs',type=int,default=10); a=ap.parse_args()
    s=json.loads((HERE/'model_spec.json').read_text()); vs=json.loads((HERE/'variants.json').read_text()); v=vs[a.variant]
    print('torch:',torch.__version__); print('cuda:',torch.version.cuda); print('cuda available:',torch.cuda.is_available()); print('gpu:',torch.cuda.get_device_name(0)); print('dataset: SemanticKITTI'); print('variant:',a.variant); print('paper_mIoU:',v['paper_mIoU'])
    m=EdgeSegmentor(s['backbone'],s['backbone_out_channels'],s['num_classes'])
    obj=torch.load(HERE/v['weight'],map_location='cpu',weights_only=False); missing,unexpected=m.load_state_dict(obj['state_dict'],strict=False)
    if missing or unexpected: print('missing:',missing); print('unexpected:',unexpected); raise SystemExit('state_dict mismatch')
    m=m.float().cuda().eval(); sm=torch.load(HERE/'sample_preprocessed.pth',map_location='cpu',weights_only=False); inp={k:sm[k].cuda() for k in ('coord','grid_coord','feat','offset')}
    n=int(inp['coord'].shape[0]); print('parameters:',sum(p.numel() for p in m.parameters())); print('processed points:',n)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(a.warmup):
            with torch.autocast(device_type='cuda',dtype=torch.float16): logits=m(inp)
        torch.cuda.synchronize(); ts=[]
        for _ in range(a.runs):
            t=time.perf_counter()
            with torch.autocast(device_type='cuda',dtype=torch.float16): logits=m(inp)
            torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1000)
    med=statistics.median(ts); mean=sum(ts)/len(ts); p95=pct(ts,.95)
    print('logits:',tuple(logits.shape),logits.dtype); print('latency_ms:',[round(x,3) for x in ts]); print('mean_ms:',round(mean,3)); print('median_ms:',round(med,3)); print('p95_ms:',round(p95,3)); print('fps_from_median:',round(1000/med,3)); print('points_per_s_from_median:',round(n/(med/1000),1)); print('peak_cuda_MiB:',round(torch.cuda.max_memory_allocated()/1024**2,1)); print('MOCHA SEMANTICKITTI JETSON SMOKE: PASS')
if __name__=='__main__': main()
