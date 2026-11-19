"""Stability-gated warmup and buffered, fixed-policy resource sampling."""
import time,statistics,threading,os,multiprocessing,json

def stable_warmup(call,sync,minimum=8,maximum=30,window=5,spread=.05,clock=time.perf_counter):
    rows=[]
    for i in range(maximum):
        sync();start=clock();call();sync();rows.append(clock()-start)
        recent=rows[-window:];median=statistics.median(recent)
        if i+1>=minimum and len(recent)==window and median>0 and (max(recent)-min(recent))/median<=spread:
            return {'stable':True,'batches':len(rows),'batch_seconds':rows,
                    'last_window_relative_range':(max(recent)-min(recent))/median,
                    'minimum':minimum,'maximum':maximum,'window':window,'allowed_relative_range':spread}
    return {'stable':False,'batches':len(rows),'batch_seconds':rows,
            'minimum':minimum,'maximum':maximum,'window':window,'allowed_relative_range':spread}


def frequency_probe_v1(sampler):
    sampler.frequency_samples=[]
    while not sampler.event.is_set():
        begin=time.perf_counter_ns()
        try:
            f=sampler.ps.cpu_freq()
            item={'monotonic_ns':begin,'MHz':f.current if f else None}
        except Exception as e:
            item={'monotonic_ns':begin,'error':repr(e)}
        item['query_seconds']=(time.perf_counter_ns()-begin)/1e9
        sampler.frequency_samples.append(item)
        sampler.latest_frequency=item
        if sampler.event.wait(1.):break

class ResourceSampler:
    def __init__(self,interval=.1):
        import pynvml,psutil
        self.n=pynvml;self.ps=psutil;self.interval=interval
        pynvml.nvmlInit();self.handles=[pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self.identity=[{'index':i,'uuid':str(pynvml.nvmlDeviceGetUUID(h)),'name':str(pynvml.nvmlDeviceGetName(h))} for i,h in enumerate(self.handles)]
        self.process=psutil.Process(os.getpid());self.process.cpu_percent(None);psutil.cpu_percent(None)
        self.samples=[];self.errors=[];self.event=threading.Event();self.thread=None;self.child_cache={}

    def sample(self):
        s=time.perf_counter_ns();gpus=[]
        for i,h in enumerate(self.handles):
            m=self.n.nvmlDeviceGetMemoryInfo(h);u=self.n.nvmlDeviceGetUtilizationRates(h)
            row={'index':i,'used_bytes':m.used,'total_bytes':m.total,'GPU_util_percent':u.gpu,'memory_util_percent':u.memory}
            for key,func,args in [('SM_clock_MHz',self.n.nvmlDeviceGetClockInfo,(h,self.n.NVML_CLOCK_SM)),
                                  ('memory_clock_MHz',self.n.nvmlDeviceGetClockInfo,(h,self.n.NVML_CLOCK_MEM)),
                                  ('power_mW',self.n.nvmlDeviceGetPowerUsage,(h,)),
                                  ('temperature_C',self.n.nvmlDeviceGetTemperature,(h,self.n.NVML_TEMPERATURE_GPU))]:
                try:row[key]=func(*args)
                except self.n.NVMLError:row[key]=None
            gpus.append(row)
        cpu=self.process.cpu_times();ctx=self.process.num_ctx_switches()
        row={'monotonic_ns':s,'gpus':gpus,
           'process_rss_bytes':self.process.memory_info().rss,'process_CPU_percent':self.process.cpu_percent(None),
           'host_CPU_percent':self.ps.cpu_percent(None),'process_CPU_user_seconds':cpu.user,
           'process_CPU_system_seconds':cpu.system,'threads':self.process.num_threads(),
           'context_switches_voluntary':ctx.voluntary,'context_switches_involuntary':ctx.involuntary}
        children=[]
        for child in self.process.children(recursive=True):
            if child.pid==os.getpid():continue
            try:
                p=self.child_cache.setdefault(child.pid,child);c=p.cpu_times();switches=p.num_ctx_switches()
                children.append({'pid':p.pid,'CPU_percent':p.cpu_percent(None),'rss_bytes':p.memory_info().rss,
                    'CPU_user_seconds':c.user,'CPU_system_seconds':c.system,'threads':p.num_threads(),
                    'voluntary_switches':switches.voluntary,'involuntary_switches':switches.involuntary})
            except (self.ps.NoSuchProcess,self.ps.AccessDenied):pass
        row['child_processes']=children
        row['CPU_frequency_observation']=getattr(self,'latest_frequency',None)
        try:row['process_CPU_affinity']=self.process.cpu_affinity()
        except Exception:row['process_CPU_affinity']=None
        row['sample_query_ns']=time.perf_counter_ns()-s
        row['sample_query_ns']=time.perf_counter_ns()-s
        self.samples.append(row)
        limit=getattr(self,'memory_limit_bytes',None)
        if limit is not None and any(g['used_bytes']>limit for g in gpus):
            flag=getattr(self,'guard_file',None)
            if flag and not os.path.exists(flag):
                with open(flag,'w',encoding='utf8') as f:json.dump({'reason':'VRAM_SAFETY_LINE','monotonic_ns':s,'limit_bytes':limit,'gpus':gpus},f)

    def start(self):
        self.sample() # mandatory monitoring capability check before hot start
        self.frequency_thread=threading.Thread(target=frequency_probe_v1,args=(self,),daemon=True)
        self.frequency_thread.start()
        def run():
            deadline=time.perf_counter()+self.interval
            while not self.event.wait(max(0,deadline-time.perf_counter())):
                try:self.sample()
                except Exception as e:self.errors.append(repr(e));return
                deadline+=self.interval
                if deadline<time.perf_counter():deadline=time.perf_counter()
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start()

    def finish(self,start_ns,end_ns):
        self.event.set();self.thread.join(timeout=3)
        if self.thread.is_alive():self.errors.append('sampler did not terminate')
        self.frequency_thread.join(timeout=3)
        if self.frequency_thread.is_alive():self.errors.append('frequency sampler did not terminate')
        rows=[s for s in self.samples if start_ns<=s['monotonic_ns']<=end_ns]
        gaps=[(b['monotonic_ns']-a['monotonic_ns'])/1e9 for a,b in zip(rows,rows[1:])]
        result={'interval_requested_seconds':self.interval,'actual_samples_during_compute':len(rows),
                'actual_gap_max_seconds':max(gaps) if gaps else None,
                'actual_gap_median_seconds':statistics.median(gaps) if gaps else None,
                'query_total_seconds':sum(s['sample_query_ns'] for s in rows)/1e9,
                'sampled_peak_is_not_continuous_peak':True,'all_physical_GPU_identities':self.identity,
                'errors':self.errors,'baseline_sample_before_hot_start':self.samples[0],
                'samples':rows,'complete':not self.errors and bool(rows)}
        result['warmup_and_startup_samples']=[s for s in self.samples if s['monotonic_ns']<start_ns]
        result['warmup_sampled_GPU_peak_bytes']={str(i):max(s['gpus'][i]['used_bytes'] for s in self.samples if s['monotonic_ns']<start_ns) for i in range(len(self.handles))} if any(s['monotonic_ns']<start_ns for s in self.samples) else {}
        boundary_gaps=([max(0,(rows[0]['monotonic_ns']-start_ns)/1e9),max(0,(end_ns-rows[-1]['monotonic_ns'])/1e9)] if rows else [(end_ns-start_ns)/1e9])
        result['largest_coverage_gap_seconds']=max(gaps+boundary_gaps)
        result['sampling_quality_pass']=result['complete'] and result['largest_coverage_gap_seconds']<=self.interval*3
        result['GPU_sampled_peak_used_bytes']={str(i):max(s['gpus'][i]['used_bytes'] for s in rows) for i in range(len(self.handles))} if rows else {}
        result['CPU_frequency_samples']=getattr(self,'frequency_samples',[])
        result['CPU_frequency_policy']='separate thread, approximately 1 second, independently timed'
        self.n.nvmlShutdown();return result

def _sample_process(interval,parent_pid,pipe,stop,result_path,memory_limit_bytes,guard_file):
    try:
        sampler=ResourceSampler(interval)
        sampler.memory_limit_bytes=memory_limit_bytes;sampler.guard_file=guard_file
        import psutil
        sampler.process=psutil.Process(parent_pid)
        sampler.process.cpu_percent(None)
        sampler.start();pipe.send({'ready':True})
        stop.wait()
        bounds=pipe.recv()
        result=sampler.finish(*bounds)
        result['sampler_in_independent_process']=True
        result['CPU_scope']='coordinator and recursive children, excluding sampler; GPU measurements cover all cards'
        with open(result_path,'w',encoding='utf8') as f:json.dump(result,f,indent=2)
    except Exception as e:
        try:pipe.send({'ready':False,'error':repr(e)})
        except Exception:pass
        raise

class ProcessResourceSampler:
    def __init__(self,result_path,interval=.1,memory_limit_bytes=None,guard_file=None):
        self.path=str(result_path);self.interval=interval
        ctx=multiprocessing.get_context('spawn')
        self.parent,self.child=ctx.Pipe();self.stop=ctx.Event()
        self.process=ctx.Process(target=_sample_process,args=(interval,os.getpid(),self.child,self.stop,self.path,memory_limit_bytes,str(guard_file) if guard_file else None),daemon=True)
    def start(self):
        self.process.start()
        if not self.parent.poll(30):raise RuntimeError('Sampler startup timed out')
        msg=self.parent.recv()
        if not msg['ready']:raise RuntimeError(str(msg))
    def finish(self,start_ns,end_ns):
        self.parent.send((start_ns,end_ns));self.stop.set();self.process.join(15)
        if self.process.is_alive():self.process.terminate();self.process.join();raise RuntimeError('Sampler exit timed out')
        if self.process.exitcode!=0:raise RuntimeError('Sampler process failed')
        with open(self.path,encoding='utf8') as f:return json.load(f)
