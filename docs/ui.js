/* Small shared presentation helpers; all data is inserted as text or SVG numbers. */
const UI = {
  $(id) { return document.getElementById(id); },
  table(id, rows) {
    const host=this.$(id); host.replaceChildren();
    if(!rows.length){host.textContent='No records.';return;}
    const keys=Object.keys(rows[0]), table=document.createElement('table'), head=table.createTHead().insertRow();
    keys.forEach(key=>{const th=document.createElement('th');th.scope='col';th.textContent=key;head.append(th);});
    const body=table.createTBody();
    rows.forEach(row=>{const tr=body.insertRow();keys.forEach(key=>{const value=row[key];tr.insertCell().textContent=value==null?'—':typeof value==='number'&&!Number.isInteger(value)?Number(value.toPrecision(5)).toString():String(value);});});
    host.append(table);
  },
  metrics(id, items) {
    const host=this.$(id);host.replaceChildren();
    items.forEach(([label,value])=>{const card=document.createElement('div'),n=document.createElement('strong'),labelNode=document.createElement('span');n.textContent=value;labelNode.textContent=label;card.append(n,labelNode);host.append(card);});
  },
  chart(id, points, title, xLabel, yLabel, baseline) {
    const ns='http://www.w3.org/2000/svg';
    const make=(name,attrs,text)=>{const e=document.createElementNS(ns,name);for(const [key,value] of Object.entries(attrs))e.setAttribute(key,String(value));if(text!=null)e.textContent=text;return e;};
    const host=this.$(id);host.chartObserver?.disconnect();let previousWidth;
    const draw=()=>{
      const width=Math.round(host.clientWidth)||700;if(width===previousWidth)return;previousWidth=width;
      const right=width-20,svg=make('svg',{viewBox:`0 0 ${width} 335`,role:'img','aria-label':title,class:'chart'});
      const x=n=>55+Math.max(0,Math.min(1,n))*(right-55),y=n=>280-Math.max(0,Math.min(1,n))*230;
      svg.append(make('text',{x:15,y:26,fill:'var(--ink)','font-size':14},title));
      for(let i=0;i<=5;i++){
        const t=i/5;
        svg.append(make('line',{x1:x(t),x2:x(t),y1:50,y2:280,stroke:'var(--line)'}),make('line',{x1:55,x2:right,y1:y(t),y2:y(t),stroke:'var(--line)'}));
        svg.append(make('text',{x:x(t),y:299,fill:'var(--muted)','font-size':11,'text-anchor':'middle'},t.toFixed(1)),make('text',{x:45,y:y(t)+4,fill:'var(--muted)','font-size':11,'text-anchor':'end'},t.toFixed(1)));
      }
      svg.append(make('line',{x1:55,y1:baseline==null?280:y(baseline),x2:right,y2:baseline==null?50:y(baseline),stroke:'var(--muted)','stroke-dasharray':'6 6'}));
      const path=points.map(([a,b],i)=>`${i?'L':'M'}${x(a).toFixed(2)},${y(b).toFixed(2)}`).join(' ');
      svg.append(make('path',{d:path,fill:'none',stroke:'var(--teal)','stroke-width':2.5}));
      svg.append(make('text',{x:(55+right)/2,y:324,fill:'var(--muted)','font-size':12,'text-anchor':'middle'},xLabel),make('text',{x:16,y:165,fill:'var(--muted)','font-size':12,transform:'rotate(-90 16 165)','text-anchor':'middle'},yLabel));
      host.replaceChildren(svg);
    };
    draw();host.chartObserver=new ResizeObserver(draw);host.chartObserver.observe(host);
  },
  evaluation(prefix,result) {
    const m=result.test_model,pct=n=>(100*n).toFixed(2)+'%',num=n=>n==null?'Undefined':n.toFixed(4);
    this.$(prefix+'-source').textContent=result.scope;
    this.$(prefix+'-note').textContent=`${result.model} · ${result.rows.train.toLocaleString()} fit rows (2014–2022), ${result.rows.validation.toLocaleString()} threshold-validation rows (2023), ${result.rows.test.toLocaleString()} historical test rows (2024). Threshold: ${result.threshold_selection.feasible?num(result.threshold_selection.threshold):'No feasible threshold; abstain'}.`;
    this.metrics(prefix+'-metrics',[['Accuracy',pct(m.accuracy)],['Balanced accuracy',pct(m.balanced_accuracy)],['Precision',pct(m.precision)],['Recall',pct(m.recall)],['F1',num(m.f1)],['ROC-AUC',num(m.roc_auc)],['Average precision',num(m.average_precision)],['False-positive rate',pct(m.false_positive_rate)]]);
    this.$(prefix+'-baseline').textContent=`Always-no-alert baseline: ${pct(result.baseline.accuracy)} accuracy, zero positive recall. Model false alerts: ${m.false_alerts.toLocaleString()}; missed positive windows: ${m.missed_events.toLocaleString()}. Scores are uncalibrated, and this is a previously inspected historical test.`;
    this.chart(prefix+'-roc',result.roc,'Receiver operating characteristic','False-positive rate','True-positive rate');
    this.chart(prefix+'-pr',result.pr,'Precision–recall curve','Recall','Precision',m.positive_rate);
    const [tn,fp,fn,tp]=m.confusion_matrix_tn_fp_fn_tp;
    this.table(prefix+'-confusion',[{'Actual class':'No high burden','Predicted no alert':tn,'Predicted alert':fp},{'Actual class':'High burden','Predicted no alert':fn,'Predicted alert':tp}]);
    this.$(prefix+'-report').textContent=result.report;
    this.$(prefix+'-content').hidden=false;
  },
  csv(rows) {
    const keys=Object.keys(rows[0]||{}),cell=value=>{let s=value==null?'':String(value);if(typeof value==='string'&&/^\s*[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
    return [keys.map(cell).join(','),...rows.map(row=>keys.map(key=>cell(row[key])).join(','))].join('\r\n');
  },
  download(content,name,type) {const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
};
