import copy
import json
from nn_meter.builder.rule_tester import config
from nn_meter.builder.backends import get_backend
from nn_meter.builder.rule_tester import RuleTester

def get_testcases(model_dir):
    """
    @params:

    model_dir: directory to save testcase models @jiahang: is it the same with "directory on host to save temporary tflite models
    
    "
    """
    config.set('model_dir', model_dir, 'ruletest')
    print(config.get_settings())

    tester = RuleTester()
    testcases = tester.generate()
    return testcases


def run_testcases(backend, testcases):
    profile_results = copy.deepcopy(testcases)
    for _, profile_result in profile_results.items():
        for _, model in profile_result.items():
            model_path = model['model']
            model['latency'] = backend.profile_model_file(model_path, model['shapes'])
    return profile_results


def get_fusionrule(profile_results):
    tester = RuleTester()
    return tester.analyze(profile_results)


if __name__ == '__main__':    

    # generate testcases
    testcases = get_testcases('/data/jiahang/test_models')

    with open('testcases.json', 'w') as fp:
        json.dump(testcases, fp, indent=4)

    with open('testcases.json', 'r') as fp:
        testcases = json.load(fp)

    # initialize backend
    backend = get_backend(
        backend = 'tflite_cpu', 
        params = {
            'MODEL_DIR': '/data/jiahang/test_models',
            'REMOTE_MODEL_DIR': '/mnt/sdcard/tflite_bench',
            'KERNEL_PATH': '/mnt/sdcard/tflite_bench/kernel.cl',
            'BENCHMARK_MODEL_PATH': '/data/local/tmp/benchmark_model_fixed_group_size',
            'DEVICE_SERIAL': '0000028e2c780e4e',
        }
    )
    
    # run testcases and collect profiling results
    profile_results = run_testcases(backend, testcases)
    import pdb; pdb.set_trace()
    # profile_results_copy = copy.deepcopy(profile_results)
    # # for _, profile_result in profile_results_copy.items():
    # #     for _, model in profile_result.items():
    # #         model_path = model['model']
    # #         model['latency'] = str(model['latency'])
    # with open('profile_results.json', 'w+') as fp:
    #     json.dump(profile_results_copy, fp)

    # determine fusion rules from profiling results
    result = get_fusionrule(profile_results)
    with open('result.json', 'w+') as fp:
        json.dump(result, fp, indent=4)
    
