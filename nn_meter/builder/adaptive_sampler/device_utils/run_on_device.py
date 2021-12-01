from .bench_utils import*
def run_on_android(modelpath,adb):
    #=======Push to device===========
    adb.push_files(modelpath, '/sdcard/')
    modelname=modelpath.split('/')[-1]


    command="taskset 70 /data/tf_benchmark/benchmark_model --num_threads=1   --warm_ups=10 --num_runs=50  --graph="+'/sdcard/'+modelname
    print(command)
    bench_str=adb.run_cmd(command)
    print(bench_str)
    std_ms,avg_ms=fetech_tf_bench_results(bench_str)

    #=======Clear device files=======
    adb.run_cmd(f'rm -rf /sdcard/'+modelname)
    return std_ms,avg_ms

def get_layer_pad_latency(bench_str):
   
    
    lines=bench_str.split('\n')
    flag=False
    
    t=0
    for line in lines:
        
        if flag:
            ops=line.strip().split('\t')
            #print(ops,len(ops))
            if len(ops)>1:
           
                avg=ops[3]
                name=ops[-1]
                #print(avg,name)
                if 'test' not in name and 'avg' not in avg :

                    t+=float(avg) 
       
        if 'Operator-wise Profiling Info' in line:
            flag=True 
        if 'Top by' in line and flag:
            flag=False
            
            

    #print(t)

    
    return t



def run_on_android_block(modelpath,adb):
    #=======Push to device===========
    adb.push_files(modelpath, '/sdcard/')
    modelname=modelpath.split('/')[-1]


    command="taskset 70 /data/tf_benchmark/benchmark_model --num_threads=1 --warm_ups=50 --num_runs=50 --graph="+'/sdcard/'+modelname
    #print(command)
    bench_str=adb.run_cmd(command)
    #print('bench_str',bench_str)
    print(bench_str)
    #std_ms,avg_ms=fetech_tf_bench_results(bench_str)
    #avg_ms=get_layer_pad_latency(bench_str)
    std_ms=0
    avg_ms=0
    

    #=======Clear device files=======
    adb.run_cmd(f'rm -rf /sdcard/'+modelname)
    return std_ms,avg_ms,bench_str