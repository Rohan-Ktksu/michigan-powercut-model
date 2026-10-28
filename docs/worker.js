const PYODIDE_URL='https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';
let runtime,busy=false,trained=false,tested=false,action='train';
self.report_progress=(fraction,text)=>self.postMessage({kind:'progress',action,fraction,text});
async function bytes(path){const response=await fetch(new URL(path,self.location.href));if(!response.ok)throw Error(`${path}: HTTP ${response.status}`);return new Uint8Array(await response.arrayBuffer());}
async function initialize(){
  if(runtime)return;
  self.report_progress(0.01,'Loading Python and scientific libraries for this browser (first visit only)');
  const {loadPyodide}=await import(PYODIDE_URL+'pyodide.mjs');
  runtime=await loadPyodide({indexURL:PYODIDE_URL,stdout:()=>{},stderr:()=>{}});
  await runtime.loadPackage(['numpy','pandas','scikit-learn']);
  runtime.FS.mkdirTree('/michigan/data');runtime.FS.chdir('/michigan');
  for(const name of ['michigan_boosted_tree.py','michigan_web_core.py','browser_runner.py'])runtime.FS.writeFile('/michigan/'+name,await bytes('runtime/'+name));
  self.report_progress(0.02,'Loading the verified, compressed browser subset');
  runtime.FS.writeFile('/michigan/site-data.json',await bytes('site-data.json'));
  runtime.FS.writeFile('/michigan/data/browser_dataset.csv.gz',await bytes('data/browser_dataset.csv.gz'));
  await runtime.runPythonAsync("import sys\nsys.path.insert(0,'/michigan')\nfrom browser_runner import browser_train,browser_test,browser_predict,browser_template");
}
self.onmessage=async({data})=>{
  if(busy)return;
  busy=true;action=data.action;
  try{
    if(action==='train'){await initialize();const result=JSON.parse(await runtime.runPythonAsync('browser_train()'));trained=true;self.postMessage({kind:'trained',result});}
    else if(action==='test'){if(!trained||tested)throw Error('Train first; testing runs once per experiment.');const result=JSON.parse(await runtime.runPythonAsync('browser_test()'));tested=true;self.postMessage({kind:'tested',result});}
    else if(action==='predict'){if(!tested)throw Error('Train and test first.');runtime.globals.set('request_json',JSON.stringify(data.input));self.postMessage({kind:'predicted',result:JSON.parse(await runtime.runPythonAsync('browser_predict(request_json)'))});}
    else if(action==='template'){if(!trained)throw Error('Train first.');self.postMessage({kind:'template',csv:await runtime.runPythonAsync('browser_template()')});}
    else throw Error('Unknown action.');
  }catch(error){if(!trained)runtime=undefined;self.postMessage({kind:'error',action,text:String(error)});}finally{busy=false;}
};
