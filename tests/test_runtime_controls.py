import json, socket, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock
import pytest, requests
from src import runtime_configuration as life, benchmark, engines
from scripts import run_remote_benchmark as cli

class Response:
    status_code=200
    def __init__(self,data=None,chunks=None):self.data=data;self.chunks=chunks
    def json(self):return self.data
    def raise_for_status(self):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_lines(self,*args,**kwargs):return iter(self.chunks)

def free_port():
    return 19011  # No listener is opened; the port probe is mocked below.

@pytest.fixture(autouse=True)
def mock_port_probe(monkeypatch):
    probe = MagicMock()
    monkeypatch.setattr(life.socket, 'socket', lambda: probe)
    return probe

@pytest.mark.parametrize('failure',[False,True])
@pytest.mark.parametrize('prefix,negative',[('enabled',False),('disabled',False),('disabled',True)])
def test_managed_launch_argv_readiness_and_cleanup(monkeypatch,failure,prefix,negative):
    process=Mock(pid=99887);process.poll.return_value=None
    launch=Mock(return_value=process);kill=Mock();monkeypatch.setattr(life.subprocess,'Popen',launch);monkeypatch.setattr(life.os,'killpg',kill)
    help_text='--enable-prefix-caching '+('--disable-prefix-caching' if negative else '--no-enable-prefix-caching')
    monkeypatch.setattr(life.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=help_text))
    monkeypatch.setattr(life.requests,'get',lambda *a,**k:Response({'data':[{'id':'local-model','max_model_len':8192}]}))
    config={'max_model_len':8192,'gpu_memory_utilization':.6,'kv_cache_dtype':'fp8','quantization':'awq','prefix_caching':prefix}
    try:
        with life.managed_vllm('local-model',f'http://localhost:{free_port()}/v1',config) as metadata:
            assert metadata['runtime_configuration_source']=='runner_launched_vllm'
            assert metadata['startup_gpu_memory_utilization']=='0.6'
            assert metadata['startup_settings_source']=='process_arguments_not_memory_measurements'
            if failure:raise RuntimeError('benchmark failed')
    except RuntimeError:assert failure
    argv=launch.call_args.args[0]
    for flag,value in [('--max-model-len','8192'),('--gpu-memory-utilization','0.6'),('--kv-cache-dtype','fp8'),('--quantization','awq')]:assert argv[argv.index(flag)+1]==value
    assert '--generation-config' in argv and argv[argv.index('--host')+1]=='127.0.0.1'
    assert ('--enable-prefix-caching' if prefix=='enabled' else '--disable-prefix-caching' if negative else '--no-enable-prefix-caching') in argv
    kill.assert_called_once_with(process.pid,life.signal.SIGTERM);process.wait.assert_called_once()

@pytest.mark.parametrize('url',['https://localhost:9000/v1','http://remote.example:9000/v1','http://localhost/v1','http://localhost:9000/bad','http://user:password@localhost:9000/v1'])
def test_remote_or_ambiguous_launch_rejected(url):
    with pytest.raises(ValueError):
        with life.managed_vllm('local-model',url,{}):pass

def test_occupied_port_not_touched(monkeypatch, mock_port_probe):
    launch=Mock();monkeypatch.setattr(life.subprocess,'Popen',launch)
    mock_port_probe.__enter__.return_value.bind.side_effect = OSError('port occupied')
    with pytest.raises(OSError):
        with life.managed_vllm('local-model',f'http://localhost:{free_port()}/v1',{}):pass
    launch.assert_not_called()

def test_early_exit_cleans_owned_process(monkeypatch):
    process=Mock(pid=99887,returncode=2);process.poll.return_value=2
    monkeypatch.setattr(life.subprocess,'Popen',lambda *a,**k:process);kill=Mock();monkeypatch.setattr(life.os,'killpg',kill)
    with pytest.raises(RuntimeError,match='exited before readiness'):
        with life.managed_vllm('local-model',f'http://localhost:{free_port()}/v1',{}):pass
    kill.assert_called_once()

def test_startup_timeout_cleans_process(monkeypatch):
    process=Mock(pid=99887);process.poll.return_value=None
    monkeypatch.setattr(life.subprocess,'Popen',lambda *a,**k:process);kill=Mock();monkeypatch.setattr(life.os,'killpg',kill)
    monkeypatch.setattr(life.requests,'get',lambda *a,**k:Response({'data':[{'id':'wrong-model'}]}))
    ticks=iter([0,10]);monkeypatch.setattr(life.time,'monotonic',lambda:next(ticks))
    with pytest.raises(RuntimeError,match='timed out'):
        with life.managed_vllm('local-model',f'http://localhost:{free_port()}/v1',{},timeout=1):pass
    kill.assert_called_once()

@pytest.mark.parametrize('flags',[
 ['--temperature','nan'],['--temperature','3'],['--seed','-1'],['--context-length','0'],
 ['--vllm-gpu-memory-utilization','0.5'],['--launch-vllm','--engine','ollama'],
 ['--context-length','4096','--engine','vllm'],['--decision-mode','--temperature','0.4'],
 ['--launch-vllm','--engine','vllm','--vllm-base-url','http://localhost:9001/v1','--vllm-gpu-memory-utilization','nan'],
 ['--ollama-base-url','http://localhost:11434/v1'],
])
def test_invalid_controls_stop_before_network(monkeypatch,flags):
    get=Mock(side_effect=AssertionError('Network must not run'));monkeypatch.setattr(cli.requests,'get',get)
    monkeypatch.setattr(sys,'argv',['run']+flags)
    with pytest.raises(SystemExit) as e:cli.main()
    assert e.value.code==2;get.assert_not_called()

def test_ollama_payload_and_observed_context(monkeypatch):
    payloads=[]
    chunks=[json.dumps({'message':{'content':'{"answer":4}'},'done':False}).encode(),json.dumps({'done':True,'prompt_eval_count':12,'prompt_eval_duration':100000000,'eval_count':5,'eval_duration':200000000}).encode()]
    monkeypatch.setattr(benchmark,'post_with_retry',lambda *a,**k:(payloads.append(k['json']) or Response(chunks=chunks)))
    monkeypatch.setattr(benchmark,'loaded_model_memory_mb',lambda *a:512)
    monkeypatch.setattr(benchmark.CLIENT,'get',lambda *a,**k:Response({'models':[{'name':'local-model','context_length':3072}]}))
    r=benchmark._measure_once('local-model','Return JSON','2+2',False,1,1,'t',True,min_ctx=2048,sampling={'temperature':.3,'seed':7},response_format='json',max_output_tokens=64)
    assert not r['error'] and r['response']=='{"answer":4}' and r['context_len']==3072
    assert payloads[0]['options']=={'temperature':.3,'seed':7,'num_predict':64,'num_ctx':2048}
    assert payloads[0]['format']=='json' and r['output_tokens']==5
    monkeypatch.setattr(benchmark.CLIENT,'get',lambda *a,**k:Response({'models':[{'name':'local-model'}]}))
    assert benchmark.loaded_model_context_length('local-model') is None

def test_vllm_payload_and_context_only_from_matching_model(monkeypatch):
    class Session:
        def get(self,*a,**k):return Response({'data':[{'id':'other','max_model_len':32000},{'id':'local-model','max_model_len':2048}]})
        def post(self,*a,**k):return Response({})
    payloads=[];chunks=[b'data: {"choices":[{"delta":{"content":"{\\"answer\\":4}"}}]}',b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5}}',b'data: [DONE]']
    monkeypatch.setattr(engines,'create_http_session',lambda **k:Session())
    monkeypatch.setattr(engines,'post_with_retry',lambda *a,**k:(payloads.append(k['json']) or Response(chunks=chunks)))
    r=engines.run_vllm('local-model','local','warm',1,store_responses=True,sampling={'temperature':.3,'seed':7},response_format='json',max_output_tokens=64,custom_tasks=[{'id':'t','prompt':'Return JSON with answer 4'}])[0]
    assert not r['error'] and r['context_len']==2048 and r['output_tokens']==5
    assert payloads[0]['response_format']=={'type':'json_object'} and payloads[0]['temperature']==.3 and payloads[0]['seed']==7
    with pytest.raises(RuntimeError):engines.run_vllm('unserved','local','warm',1,custom_tasks=[{'id':'t','prompt':'answer'}])

def test_runtime_configuration_changes_identity(monkeypatch):
    monkeypatch.setattr(benchmark,'model_metadata',lambda *a:{})
    monkeypatch.setattr(benchmark,'runtime_metadata',lambda:{})
    row={'run_index':1,'ttft_s':.1,'e2e_s':.2,'prefill_tps':100,'decode_tps':20,'prompt_tokens':5,'output_tokens':2,'model_memory_mb':512,'error':None,'context_len':2048}
    a=benchmark.summarise('local-model','local','warm',[row],engine='ollama',engine_config={'temperature':'0.3','response_format':'text'})
    b=benchmark.summarise('local-model','local','warm',[row],engine='ollama',engine_config={'temperature':'0.7','response_format':'text'})
    assert a['pipeline_config_id']!=b['pipeline_config_id']

def test_ollama_quantization_uses_engine_metadata(monkeypatch):
    monkeypatch.setattr(benchmark.CLIENT,'get',lambda *a,**k:Response({'models':[{'name':'local-model','digest':'sha256:local','size':123}]}))
    monkeypatch.setattr(benchmark.CLIENT,'post',lambda *a,**k:Response({'details':{'quantization_level':'Q4_K_M'}}))
    metadata=benchmark.model_metadata('local-model')
    assert metadata['quantization']=='Q4_K_M' and metadata['digest']=='sha256:local'
