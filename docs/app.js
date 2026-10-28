const $=id=>UI.$(id);
let worker,busy=false,trained=false,tested=false,lastResult,predictions=[];
function error(text){$('error').hidden=false;$('error').textContent=text;}
function buttons(){$('train').disabled=busy||trained;$('test').disabled=busy||!trained||tested;$('prediction-controls').disabled=busy||!tested;}
function start(action,input){if(busy)return;busy=true;buttons();$('error').hidden=true;if(['train','test'].includes(action))$(action+'-progress').hidden=false;if(action==='predict'){$('prediction-output').hidden=true;$('predict-status').textContent='Predicting from the fitted model…';}worker.postMessage({action,input});}
async function init(){
  const response=await fetch('site-data.json');if(!response.ok)throw Error('Dataset preview unavailable. Refresh to retry.');
  const data=await response.json();UI.table('data-preview',data.preview);data.counties.forEach(name=>$('county').add(new Option(name,name)));
  if(!window.Worker||!window.WebAssembly)throw Error('Use a current browser supporting WebAssembly and Web Workers.');
  worker=new Worker('worker.js',{type:'module'});
  worker.onerror=event=>{busy=false;buttons();error('The Python worker stopped: '+event.message+'. Reload to start a new experiment.');};
  worker.onmessage=({data})=>{
    if(data.kind==='progress'){$(data.action+'-status').textContent=data.text;$(data.action+'-bar').value=data.fraction;if(data.action==='train'){const li=document.createElement('li');li.textContent=data.text;$('train-log').append(li);li.scrollIntoView({block:'nearest'});}return;}
    busy=false;
    if(data.kind==='error'){error(data.text);if(['train','test'].includes(data.action))$(data.action+'-status').textContent='Failed. See the message below.';buttons();return;}
    if(data.kind==='trained'){
      trained=true;$('train-badge').textContent='TRAINING COMPLETE';$('train-step').classList.add('current');$('training-output').hidden=false;
      const s=data.result.threshold_selection;
      $('training-output').textContent=s.feasible?`180 iterations complete. Validation threshold: ${s.threshold.toFixed(5)} · validation FPR ${(s.validation_false_positive_rate*100).toFixed(2)}%. Ready to test.`:'No threshold met the validation constraints. The model will abstain from issuing alerts; testing will show that outcome.';
      $('runtime-note').textContent=`Python ${data.result.python} · scikit-learn ${data.result.sklearn} · 67 inputs · 20,000 fit / 4,000 validation / 5,000 test rows. Browser sample results can differ from the full published model.`;
      $('testing').classList.remove('locked');$('test-badge').textContent='READY TO TEST';
    }
    if(data.kind==='tested'){
      tested=true;lastResult={...data.result,evaluated_at:new Date().toISOString()};UI.evaluation('test',lastResult);
      try{sessionStorage.setItem('michigan-evaluation',JSON.stringify(lastResult));}catch{$('runtime-note').textContent+=' Browser storage is unavailable; Model & Features will show the published full-research run.';}
      $('test-step').classList.add('current');$('test-badge').textContent='TEST COMPLETE';$('prediction').classList.remove('locked');$('predict-badge').textContent='READY TO PREDICT';
    }
    if(data.kind==='predicted'){
      predictions=data.result.results;$('predict-step').classList.add('current');$('predict-status').textContent=`Prediction complete · ${predictions.length} historical records`;$('prediction-output').hidden=false;
      UI.metrics('prediction-metrics',[['High-burden alerts',predictions.filter(r=>r.high_burden_alert).length],['No-alert outputs',predictions.filter(r=>!r.high_burden_alert).length],['Records',predictions.length],['Horizon','3 hours']]);
      UI.table('predictions',predictions);UI.table('prediction-inputs',data.result.inputs);
    }
    if(data.kind==='template')UI.download(data.csv,'michigan_input_template.csv','text/csv');buttons();
  };buttons();
}
$('train').onclick=()=>start('train');$('test').onclick=()=>start('test');$('template').onclick=()=>start('template');
$('predict').onclick=async()=>{
  if($('prediction-source').value==='upload'){const file=$('upload').files[0];if(!file)return error('Choose a CSV first.');if(file.size>5*1024*1024)return error('Use a CSV of at most 5 MB.');start('predict',{csv:await file.text()});}
  else{const count=Number($('count').value);if(!Number.isInteger(count)||count<1||count>100)return error('Choose 1–100 records.');start('predict',{county:$('county').value,count});}
};
$('download-results').onclick=()=>UI.download(JSON.stringify(lastResult,null,2),'michigan_evaluation.json','application/json');
$('download-predictions').onclick=()=>UI.download(UI.csv(predictions),'michigan_predictions.csv','text/csv');
for(const id of ['prediction-source','county','count','upload'])$(id).onchange=()=>{$('prediction-output').hidden=true;$('predict-status').textContent='';const upload=$('prediction-source').value==='upload';$('upload-box').hidden=!upload;$('county-label').hidden=upload;$('count-label').hidden=upload;};
init().catch(e=>error(e.message));
