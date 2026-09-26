"""Field, event and paired-response losses for the current model."""
from typing import Sequence, Optional
import numpy as np
import torch
from torch.nn import functional as F
from model.sp_jepa import sigreg_v2


def masked_field_mse(pred, target, valid, area=None):
    """Per sample/branch/lead/channel reduction followed by equal valid-item mean."""
    if pred.shape != target.shape or valid.shape != target.shape: raise ValueError("field tensors disagree")
    w=valid.to(pred.dtype)
    if area is not None:
        a=area.to(pred.dtype)
        while a.ndim < w.ndim: a=a.unsqueeze(1)
        w=w*a
    num=((pred-target).square()*w).sum((-2,-1)); den=w.sum((-2,-1))
    return (num/den.clamp_min(1)).masked_select(den>0).mean() if (den>0).any() else pred.sum()*0



def event_loss(raw, truth, available, profile):
    """Mean available families within each lead, then mean supervised leads."""
    if profile == "paired_transport_exchange":
        pred=raw.reshape(*raw.shape[:-1],2,2); target=truth.reshape_as(pred)
        per=.5*(pred-target).square().sum(-1)
    else:
        endpoints=raw[...,:10].reshape(*raw.shape[:-1],5,2).sigmoid().sort(-1).values
        t_end=truth[...,:10].reshape_as(endpoints)
        endpoint_loss=.5*(endpoints-t_end).square().sum(-1)
        cls=F.cross_entropy(raw[...,10:].reshape(-1,3),truth[...,10:].argmax(-1).reshape(-1),reduction="none")
        per=torch.cat((endpoint_loss,cls.reshape(raw.shape[:-1])[...,None]),-1)
    if per.shape != available.shape or per.ndim < 2:
        raise ValueError("event loss requires [...,lead,family] availability mask")
    # Collapse only sample/branch axes.  A sparsely known lead-family cell must
    # not receive less weight merely because it has fewer known samples.
    per=per.reshape(-1,per.shape[-2],per.shape[-1])
    available=available.bool().reshape_as(per)
    count=available.sum(0)
    cell=(per*available).sum(0)/count.clamp_min(1)
    supervised=count>0
    family_count=supervised.sum(-1)
    lead=(cell*supervised).sum(-1)/family_count.clamp_min(1)
    return lead.masked_select(family_count>0).mean() if (family_count>0).any() else raw.sum()*0



def fixed_q(fields, events, event_known=None, event_scale=None):
    """Paired q: two full standardized fields and two event values, group weights 1/2."""
    f=fields.flatten(-3)
    # grid/channel equal: scaling by sqrt(size) makes squared L2 the group MSE.
    f=f/(2*fields.shape[-2]*fields.shape[-1])**.5
    if events is None: return f/(2**.5), None
    e=events
    if event_scale is not None: e=e/event_scale
    e=e/(events.shape[-1]**.5)
    if event_known is not None: e=torch.where(event_known,e,torch.zeros_like(e))
    return f/(2**.5),e/(2**.5)



def response_loss_parts(pred_f, true_f0, true_fu, pred_f0, pred_e=None, true_e0=None, true_eu=None,
                        pred_e0=None, event_joint=None, event_scale=None):
    """Field and event groups of the fixed-q signed response error; their sum is response_loss."""
    pf,pe=fixed_q(pred_f-pred_f0, None if pred_e is None else pred_e-pred_e0, event_joint,event_scale)
    tf,te=fixed_q(true_fu-true_f0, None if true_eu is None else true_eu-true_e0,event_joint,event_scale)
    field=(pf-tf).square().sum(-1)
    event=torch.zeros_like(field) if pe is None else (pe-te).square().sum(-1)
    # Missing event gives zero contribution while field remains at 1/2 through q scaling.
    return field.mean(), event.mean()



def response_loss(*args, **kwargs):
    field,event=response_loss_parts(*args, **kwargs)
    return field+event



def response_event_truth(batch):
    """Point-identified factual/changed event truths and their joint mask (paired environment)."""
    event_joint=batch["future_event_known"].bool() & batch["changed_event_known"].bool()
    factual_truth_pairs=batch["future_event_truth"].reshape(*batch["future_event_truth"].shape[:-1],2,2)
    changed_truth_pairs=batch["changed_event_truth"].reshape(*batch["changed_event_truth"].shape[:-1],2,2)
    # A point-identified target has equal endpoints; never form an unidentified midpoint.
    true_factual=torch.where(event_joint,factual_truth_pairs[...,0],torch.zeros_like(factual_truth_pairs[...,0]))
    true_changed=torch.where(event_joint,changed_truth_pairs[...,0],torch.zeros_like(changed_truth_pairs[...,0]))
    return true_factual,true_changed,event_joint



@torch.no_grad()
def true_response_energy(batch):
    """E||q(r*)||^2 with the fixed q/W of response_loss (field and event groups summed)."""
    zero_f=torch.zeros_like(batch["target"])
    args={}
    if "changed_event_truth" in batch:
        true_factual,true_changed,event_joint=response_event_truth(batch)
        zero_e=torch.zeros_like(true_factual)
        args={"pred_e":zero_e,"true_e0":true_factual,"true_eu":true_changed,"pred_e0":zero_e,
              "event_joint":event_joint,"event_scale":batch["event_scale"]}
    field,event=response_loss_parts(zero_f,batch["target"],batch["changed_target"],zero_f,**args)
    return float(field+event)



def world_losses(model, batch, lambda_sig=.09, lambda_event=.02, lambda_response=.1, sigreg_directions=1024,
                 response_scale=None):
    """World objective split into the data term L_D and the event term L_V (METHOD 5.4).

    ``response_scale`` is the frozen TRAIN response energy V_train; when given,
    the response term is divided by it (energy-normalized form).  ``total`` is
    always ``data_total + event_total``.
    """
    sigreg_keys=("sigreg_x","sigreg_valid","sigreg_geometry","sigreg_calendar",
                 "sigreg_target","sigreg_target_valid","sigreg_future_calendar")
    missing=[key for key in sigreg_keys if key not in batch]
    if missing:
        raise ValueError(f"explicit B=8 SIGReg batch required; missing {missing}")
    if model.spec.profile == "paired_transport_exchange" and "future_event_truth" in batch:
        boundary_keys=["future_event_boundary_valid"]
        if "current_event_truth" in batch: boundary_keys.append("current_event_boundary_valid")
        if "changed_event_truth" in batch: boundary_keys.append("changed_event_boundary_valid")
        if "changed_current_event_truth" in batch: boundary_keys.append("changed_current_event_boundary_valid")
        missing_boundary=[key for key in boundary_keys if key not in batch]
        if missing_boundary:
            raise ValueError(f"Paired endpoint supervision requires boundary availability; missing {missing_boundary}")
    state=model.encode(batch["x"],batch["valid"],batch["geometry"],batch["calendar"])
    factual=model.rollout(state,batch["calendar"],batch["future_calendar"])
    factual_fields=model.read_fields(factual,batch["geometry"])
    target_state=model.encode(batch["target"],batch["target_valid"],batch["geometry"],batch["future_calendar"])
    jepa=F.mse_loss(factual["h"],target_state["h"])
    field=masked_field_mse(factual_fields,batch["target"],batch["target_valid"],batch.get("area"))
    # The sampler supplies exactly one factual root per independent unit.
    ss=model.encode(batch["sigreg_x"],batch["sigreg_valid"],batch["sigreg_geometry"],batch["sigreg_calendar"])
    st=model.encode(batch["sigreg_target"],batch["sigreg_target_valid"],batch["sigreg_geometry"],batch["sigreg_future_calendar"])
    sig_h=torch.cat((ss["h"],st["h"]),1)
    sig=sigreg_v2(sig_h,directions=sigreg_directions)
    eraw=factual["events"]
    if "future_event_truth" in batch:
        future_available=(batch["future_event_boundary_valid"] if model.spec.profile == "paired_transport_exchange"
                          else batch["future_event_known"])
        if "current_event_truth" in batch:
            eraw=torch.cat((model.read_events(state)[:,None],eraw),1)
            et=torch.cat((batch["current_event_truth"][:,None],batch["future_event_truth"]),1)
            current_available=(batch["current_event_boundary_valid"] if model.spec.profile == "paired_transport_exchange"
                               else batch["current_event_known"])
            available=torch.cat((current_available[:,None],future_available),1)
        else: et,available=batch["future_event_truth"],future_available
        ev=event_loss(eraw,et,available,model.spec.profile)
    else: ev=eraw.sum()*0
    response=field.sum()*0; response_field=response; response_event=response
    outputs={"state":state,"factual":factual,"factual_fields":factual_fields,"target_state":target_state}
    if model.spec.profile == "paired_transport_exchange" and "changed_target" in batch:
        edited=model.edit(state,batch["action"])
        changed=model.rollout(edited,batch["calendar"],batch["future_calendar"])
        changed_fields=model.read_fields(changed,batch["geometry"])
        ct=model.encode(batch["changed_target"],batch["changed_target_valid"],batch["geometry"],batch["future_calendar"])
        jepa=.5*(jepa+F.mse_loss(changed["h"],ct["h"]))
        field=.5*(field+masked_field_mse(changed_fields,batch["changed_target"],batch["changed_target_valid"],batch.get("area")))
        response_args={}
        if "changed_event_truth" in batch:
            if "event_scale" not in batch:
                raise ValueError("joint response loss requires frozen event_scale")
            factual_events=factual["events"].reshape(*factual["events"].shape[:-1],2,2).mean(-1)
            changed_events=changed["events"].reshape(*changed["events"].shape[:-1],2,2).mean(-1)
            true_factual_events,true_changed_events,event_joint=response_event_truth(batch)
            response_args={"pred_e":changed_events,"true_e0":true_factual_events,"true_eu":true_changed_events,
                           "pred_e0":factual_events,"event_joint":event_joint,
                           "event_scale":batch["event_scale"]}
        response_field,response_event=response_loss_parts(changed_fields,batch["target"],batch["changed_target"],factual_fields,**response_args)
        if response_scale is not None:
            if not float(response_scale) > 0: raise ValueError("response_scale must be a positive frozen TRAIN energy")
            response_field=response_field/float(response_scale); response_event=response_event/float(response_scale)
        response=response_field+response_event
        if "changed_event_truth" in batch:
            ceraw=changed["events"]; cet=batch["changed_event_truth"]; changed_available=batch["changed_event_boundary_valid"]
            if "changed_current_event_truth" in batch:
                ceraw=torch.cat((model.read_events(edited)[:,None],ceraw),1)
                cet=torch.cat((batch["changed_current_event_truth"][:,None],cet),1)
                changed_available=torch.cat((batch["changed_current_event_boundary_valid"][:,None],changed_available),1)
            ev=.5*(ev+event_loss(ceraw,cet,changed_available,model.spec.profile))
        outputs.update(changed=changed,changed_fields=changed_fields,changed_target_state=ct)
    data_total=jepa+lambda_sig*sig+field+lambda_response*response_field
    event_total=lambda_event*ev+lambda_response*response_event
    total=data_total+event_total
    return {"total":total,"data_total":data_total,"event_total":event_total,"jepa":jepa,"sigreg":sig,
            "field":field,"event":ev,"response":response,"response_field":response_field,
            "response_event":response_event,**outputs}



def frozen_pair_table(features: np.ndarray, root_ids: Sequence[int], mechanism_ids: Optional[Sequence[int]]=None, neighbors=4):
    x=np.asarray(features,float); ids=np.asarray(root_ids,int)
    if len(x)!=len(ids) or len(set(ids.tolist()))!=len(ids): raise ValueError("root ids invalid")
    mechanisms=np.zeros(len(ids),int) if mechanism_ids is None else np.asarray(mechanism_ids,int)
    edges=[]
    for i in range(len(ids)):
        ok=np.where((mechanisms==mechanisms[i]) & (np.arange(len(ids))!=i))[0] if mechanism_ids is not None else np.where(np.arange(len(ids))!=i)[0]
        dist=((x[ok]-x[i])**2).mean(tuple(range(1,x.ndim)))
        for j in ok[np.argsort(dist)[:neighbors]]:
            a,b=sorted((i,int(j))); edges.append((float(((x[a]-x[b])**2).mean()),int(ids[a]),int(ids[b]),int(mechanisms[a])))
    unique={(a,b):(d,a,b,m) for d,a,b,m in edges}.values()
    threshold=np.percentile([e[0] for e in unique],25)
    used=set(); table=[]
    for e in sorted((e for e in unique if e[0]<=threshold),key=lambda z:(z[0],z[1],z[2])):
        if e[1] not in used and e[2] not in used: table.append(e); used|={e[1],e[2]}
    if len(table)<16: raise RuntimeError("insufficient_pairs")
    if mechanism_ids is not None and len({e[3] for e in table})<8: raise RuntimeError("insufficient_pairs")
    return table,float(threshold)



CORE_MODULES={"E":("encoder",),
              "F":("a_in","cal_in","time_pos","blocks","event_proj","calendar_proj","h_out","a_out")}



def core_parameter_groups(model):
    """Trainable E and F parameters of SPJEPA (METHOD 5.4 projection modules)."""
    groups={}
    for name,prefixes in CORE_MODULES.items():
        params=[p for n,p in model.named_parameters()
                if p.requires_grad and any(n==prefix or n.startswith(prefix+".") for prefix in prefixes)]
        if not params: raise ValueError(f"model has no trainable {name} parameters")
        groups[name]=params
    return groups



def data_priority_backward(model, data_loss, event_loss):
    """Set .grad to g_D+g_V, removing the part of g_V that conflicts with g_D on E and F.

    For each core module m: g_m = g_D + g_V - min(0,<g_V,g_D>)/||g_D||^2 g_D.
    Non-core parameters (D_x, G_e) receive their plain gradients.  Returns the
    per-module cosine between g_V and g_D before projection (NaN if undefined).
    """
    params=[p for p in model.parameters() if p.requires_grad]
    fill=lambda grads:[torch.zeros_like(p) if g is None else g for p,g in zip(params,grads)]
    g_data=fill(torch.autograd.grad(data_loss,params,retain_graph=True,allow_unused=True))
    g_event=fill(torch.autograd.grad(event_loss,params,allow_unused=True))
    index={id(p):i for i,p in enumerate(params)}
    for p,gd,gv in zip(params,g_data,g_event): p.grad=gd+gv
    report={}
    for name,group in core_parameter_groups(model).items():
        ids=[index[id(p)] for p in group]
        dot=sum((g_event[i]*g_data[i]).sum() for i in ids)
        norm_d=sum(g_data[i].square().sum() for i in ids)
        norm_v=sum(g_event[i].square().sum() for i in ids)
        cosine=float(dot/(norm_d.sqrt()*norm_v.sqrt())) if norm_d>0 and norm_v>0 else float("nan")
        report[f"projection_cos_{name}"]=cosine
        if dot<0 and norm_d>0:
            scale=dot/norm_d
            for i in ids: params[i].grad=g_data[i]+g_event[i]-scale*g_data[i]
    return report
