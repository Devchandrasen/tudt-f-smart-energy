"""Vector architecture schematic, with controller/evaluator timing explicit."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'results/audit_v2/figures'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':7.8,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(7.15,3.65))
        fig.subplots_adjust(0,0,1,1);ax.set_xlim(0,18);ax.set_ylim(0,9.2);ax.axis('off')
        ink='#183247';blue='#215E8C';teal='#227B78';muted='#526574';line='#718697'
        def box(x,y,w,h,title,lines=(),fill='#F2F6FA',edge=blue,fs=7.0):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.045,rounding_size=.12',linewidth=.85,edgecolor=edge,facecolor=fill))
            ax.text(x+.16,y+h-.24,title,ha='left',va='top',weight='bold',color=edge,fontsize=8.2)
            for j,s in enumerate(lines):ax.text(x+.16,y+h-.79-.40*j,s,ha='left',va='top',color=ink,fontsize=fs)
        def arrow(points,color=line,dashed=False,lw=.95):
            for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,ls='--' if dashed else '-')
            ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=8.5,lw=lw,color=color,linestyle='--' if dashed else '-'))
        ax.text(.25,8.93,'CONTROLLER',fontsize=8.8,weight='bold',color=blue)
        ax.text(3.1,8.93,'Only information available at decision time t',fontsize=8.1,color=muted)
        box(.25,5.65,3.05,2.85,'1  Semantic state',[
            'Load / generation','Time / availability','Posted price / SOC','Typed observations'])
        box(3.85,5.65,3.12,2.85,'2  Prediction',[
            'Past-fitted forest','Load / generation',r'$\widehat{y}_{t+1\mid t}$','Stored forecasts'])
        box(7.52,5.65,3.3,2.85,'3  Uncertainty',[
            'Forecast intervals','Split / block / adaptive','Width / past coverage','Nominal target: 0.9'])
        arrow([(3.3,7.1),(3.85,7.1)])
        arrow([(6.97,7.1),(7.52,7.1)])
        # This path is a proposal input, not a use of future physical truth.
        arrow([(5.40,8.50),(5.40,8.73),(14.5,8.73),(14.5,8.13)])
        ax.text(11.1,8.77,'Forecast + price / SOC',fontsize=6.9,color=muted,va='bottom')
        box(3.85,1.86,6.97,3.05,'4  Observable trust evidence',fill='#F0F8F7',edge=teal)
        channels=[(4.03,3.30,'TSS','Age / past residual'),(7.44,3.30,'DRS','Complete / rough / valid'),
                  (4.03,2.28,'MUS','Width / past coverage'),(7.44,2.28,'SAS','Jump / residual / valid')]
        for x,y,title,detail in channels:
            ax.add_patch(FancyBboxPatch((x,y),3.19,.79,boxstyle='round,pad=.025,rounding_size=.07',facecolor='white',edgecolor='#91B6B0',linewidth=.6))
            ax.text(x+.10,y+.68,title,color=teal,weight='bold',fontsize=7.5,va='top')
            ax.text(x+.10,y+.24,detail,color=ink,fontsize=6.3)
        ax.text(4.03,2.01,'Shared signals are dependent evidence',fontsize=6.6,color=muted)
        arrow([(2.95,5.65),(2.95,3.40),(3.85,3.40)],color=teal)
        ax.text(.40,4.55,'Received values',fontsize=6.0,color=teal,ha='left')
        ax.text(.40,4.19,'and timestamps',fontsize=6.0,color=teal,ha='left')
        arrow([(5.39,5.65),(5.39,4.91)],color=teal)
        ax.text(5.63,5.20,'Past pairs (s ≤ t)',fontsize=6.65,color=muted)
        arrow([(9.18,5.65),(9.18,4.91)],color=teal)
        # Component 5 exposes proposal, policy and physical check separately.
        box(11.55,1.86,6.19,6.26,'5  Authorization and dispatch',fill='#F6F7FA',edge=blue)
        ax.add_patch(FancyBboxPatch((11.79,6.51),5.71,.85,boxstyle='round,pad=.03,rounding_size=.07',facecolor='white',edgecolor='#9AAFC0',linewidth=.6))
        ax.text(14.64,7.06,r'Task-specific candidate $u_t^\star$',weight='bold',ha='center',color=ink,fontsize=7.7)
        ax.text(14.64,6.72,'Battery dispatch / PV curtailment',ha='center',color=muted,fontsize=6.75)
        arrow([(14.64,6.5),(14.64,5.94)])
        ax.text(14.64,5.60,r'$T_t=\sum_j w_j S_t^j$  and frozen thresholds',ha='center',color=ink,fontsize=8.1)
        modes=[(11.82,'FULL',r'$T_t\geq\tau_h$'),(13.76,'SCALED',r'$\tau_l\leq T_t<\tau_h$'),(15.70,'DEFER',r'$T_t<\tau_l$')]
        for x,title,detail in modes:
            ax.add_patch(FancyBboxPatch((x,4.31),1.69,.93,boxstyle='round,pad=.035,rounding_size=.07',facecolor='#E7EEF5',edgecolor='#9AAFC0',linewidth=.55))
            ax.text(x+.845,4.92,title,ha='center',weight='bold',fontsize=7.0,color=blue)
            ax.text(x+.845,4.55,detail,ha='center',fontsize=7.1,color=ink)
        arrow([(10.82,3.41),(11.19,3.41),(11.19,5.52),(11.55,5.52)],color=teal)
        ax.text(11.0,3.55,r'$T_t$',ha='center',fontsize=7.2,color=teal)
        arrow([(14.64,4.28),(14.64,3.80)])
        ax.add_patch(FancyBboxPatch((11.79,2.46),5.71,1.29,boxstyle='round,pad=.03,rounding_size=.07',facecolor='white',edgecolor='#9AAFC0',linewidth=.65))
        ax.text(14.64,3.45,'Recheck the issued command',ha='center',weight='bold',fontsize=7.6,color=ink)
        ax.text(14.64,3.06,'Energy limits + forecast AC feasibility',ha='center',fontsize=7.0,color=ink)
        ax.text(14.64,2.68,'Apply fallback; retain review reason',ha='center',fontsize=6.6,color=muted)
        ax.text(14.64,2.08,'Log candidate, mode, command and feasibility',ha='center',fontsize=6.6,color=muted)
        # Evaluator has no arrow into the controller.
        ax.plot([.25,17.74],[1.48,1.48],color='#9D574A',lw=.9,ls=(0,(4,3)))
        ax.text(.30,1.60,'INFORMATION BOUNDARY',color='#8B4439',fontsize=7.0,weight='bold')
        ax.add_patch(FancyBboxPatch((.25,.16),17.49,1.01,boxstyle='round,pad=.04,rounding_size=.10',facecolor='#FBF3EF',edgecolor='#BE8C7B',linewidth=.75))
        ax.text(.43,.89,'EVALUATOR  |  next-hour observations',fontsize=7.7,weight='bold',color='#804537')
        ax.text(.43,.49,'Uncorrupted outcomes + injection labels',fontsize=7.2,color=ink)
        ax.text(8.10,.89,'Error exposure / deferral / cost / realized violations',fontsize=7.2,color=ink)
        ax.text(8.10,.49,'Includes holds and fallback commands; no evaluator feedback to the gate',fontsize=6.7,color=muted)
        arrow([(17.36,1.85),(17.36,1.18)],color='#8B4439')
        for ext in ['pdf','svg','png']:
            fig.savefig(DEST/f'architecture_v4.{ext}',dpi=600,facecolor='white',transparent=False)
        plt.close(fig)
    (DEST/'architecture_v4_provenance.json').write_text(json.dumps({'type':'architecture schematic','empirical_data_transformation':'none','width_inches':7.15,'height_inches':3.65,'source':['data_pipeline.py','control_v2.py','manuscript_v2/body.tex'],'purpose':'Five functional components, dependent trust inputs, issued-command recheck and evaluator timing'},indent=2))
    print(DEST/'architecture_v4.pdf')

if __name__=='__main__':main()
