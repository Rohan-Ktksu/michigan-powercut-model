async function initModel(){
  const response=await fetch('site-data.json');if(!response.ok)throw Error('Feature schema unavailable.');
  const data=await response.json();UI.table('features',data.schema);UI.$('excluded').textContent=data.excluded.join(', ');
  const groups=[...new Set(data.schema.map(row=>row.group))];
  groups.forEach(group=>{const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent=group;pre.className='plain-code';pre.textContent=data.schema.filter(row=>row.group===group).map(row=>row.feature).join(', ');details.append(summary,pre);UI.$('feature-groups').append(details);});
  let result;try{result=JSON.parse(sessionStorage.getItem('michigan-evaluation'));}catch{}
  if(!result?.test_model||!result?.roc){const saved=await fetch('evaluation.json');if(!saved.ok)throw Error('Evaluation unavailable.');result=await saved.json();}
  UI.evaluation('saved',result);
}
initModel().catch(e=>{UI.$('saved-note').textContent=e.message;});
