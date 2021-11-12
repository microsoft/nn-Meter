(JUST AN EXAMPLE FROM NNI, NEED REFINE. JIAHANG)

# Register customized backend as builtin modules

NNI provides state-of-the-art tuning algorithm in builtin-tuners. NNI supports to build a tuner by yourself for tuning demand.

If you want to implement your own tuning algorithm, you can implement a customized Tuner, there are three things to do:

- Inherit the base Tuner class

- Implement receive_trial_result, generate_parameter and update_search_space function

- Configure your customized tuner in experiment YAML config file

Here is an example:

You can follow below steps to build a customized backend, and register it into nn-Meter as builtin modules.

### 1. Inherit the base Tuner class

```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...
```

2. Implement receive_trial_result, generate_parameter and update_search_space function

```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Receive trial's final result.
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        value: final metrics of the trial, including default metric
        '''
        # your code implements here.
    ...

    def generate_parameters(self, parameter_id, **kwargs):
        '''
        Returns a set of trial (hyper-)parameters, as a serializable object
        parameter_id: int
        '''
        # your code implements here.
        return your_parameters
    ...

    def update_search_space(self, search_space):
        '''
        Tuners are advised to support updating search space at run-time.
        If a tuner can only set search space once before generating first hyper-parameters,
        it should explicitly document this behaviour.
        search_space: JSON object created by experiment owner
        '''
        # your code implements here.
    ...
```
receive_trial_result will receive the parameter_id, parameters, value as parameters input. Also, Tuner will receive the value object are exactly same value that Trial send.

The your_parameters return from generate_parameters function, will be package as json object by NNI SDK. NNI SDK will unpack json object so the Trial will receive the exact same your_parameters from Tuner.

For example: If the you implement the generate_parameters like this:
```python
def generate_parameters(self, parameter_id, **kwargs):
    '''
    Returns a set of trial (hyper-)parameters, as a serializable object
    parameter_id: int
    '''
    # your code implements here.
    return {"dropout": 0.3, "learning_rate": 0.4}
```

It means your Tuner will always generate parameters {"dropout": 0.3, "learning_rate": 0.4}. Then Trial will receive {"dropout": 0.3, "learning_rate": 0.4} by calling API nni.get_next_parameter(). Once the trial ends with a result (normally some kind of metrics), it can send the result to Tuner by calling API nni.report_final_result(), for example nni.report_final_result(0.93). Then your Tuner’s receive_trial_result function will receied the result like：
```python
parameter_id = 82347
parameters = {"dropout": 0.3, "learning_rate": 0.4}
value = 0.93
```
Note that The working directory of your tuner is <home>/nni-experiments/<experiment_id>/log, which can be retrieved with environment variable NNI_LOG_DIRECTORY, therefore, if you want to access a file (e.g., data.txt) in the directory of your own tuner, you cannot use open('data.txt', 'r'). Instead, you should use the following:
```python
_pwd = os.path.dirname(__file__)
_fd = open(os.path.join(_pwd, 'data.txt'), 'r')
```
This is because your tuner is not executed in the directory of your tuner (i.e., pwd is not the directory of your own tuner).

3. Configure your customized tuner in experiment YAML config file

NNI needs to locate your customized tuner class and instantiate the class, so you need to specify the location of the customized tuner class and pass literal values as parameters to the __init__ constructor.
```python
tuner:
  codeDir: /home/abc/mytuner
  classFileName: my_customized_tuner.py
  className: CustomizedTuner
  # Any parameter need to pass to your tuner class __init__ constructor
  # can be specified in this optional classArgs field, for example
  classArgs:
    arg1: value1
```
More detail example you could see:


Write a more advanced automl algorithm
The methods above are usually enough to write a general tuner. However, users may also want more methods, for example, intermediate results, trials’ state (e.g., the methods in assessor), in order to have a more powerful automl algorithm. Therefore, we have another concept called advisor which directly inherits from MsgDispatcherBase in msg_dispatcher_base.py. Please refer to here for how to write a customized advisor.



5. Register customized algorithms into NNI

Run following command to register the customized algorithms as builtin algorithms in NNI:

nnictl algo register --meta <path_to_meta_file>
The <path_to_meta_file> is the path to the yaml file your created in above section.

Reference customized tuner example for a full example.

Use the installed builtin algorithms in experiment
Once your customized algorithms is installed, you can use it in experiment configuration file the same way as other builtin tuners/assessors/advisors, for example:

tuner:
  builtinTunerName: demotuner
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
Manage builtin algorithms using nnictl algo
List builtin algorithms
Run following command to list the registered builtin algorithms:

nnictl algo list
+-----------------+------------+-----------+--------=-------------+------------------------------------------+
|      Name       |    Type    | Source    |      Class Name      |               Module Name                |
+-----------------+------------+-----------+----------------------+------------------------------------------+
| TPE             | tuners     | nni       | HyperoptTuner        | nni.hyperopt_tuner.hyperopt_tuner        |
| Random          | tuners     | nni       | HyperoptTuner        | nni.hyperopt_tuner.hyperopt_tuner        |
| Anneal          | tuners     | nni       | HyperoptTuner        | nni.hyperopt_tuner.hyperopt_tuner        |
| Evolution       | tuners     | nni       | EvolutionTuner       | nni.evolution_tuner.evolution_tuner      |
| BatchTuner      | tuners     | nni       | BatchTuner           | nni.batch_tuner.batch_tuner              |
| GridSearch      | tuners     | nni       | GridSearchTuner      | nni.gridsearch_tuner.gridsearch_tuner    |
| NetworkMorphism | tuners     | nni       | NetworkMorphismTuner | nni.networkmorphism_tuner.networkmo...   |
| MetisTuner      | tuners     | nni       | MetisTuner           | nni.metis_tuner.metis_tuner              |
| GPTuner         | tuners     | nni       | GPTuner              | nni.gp_tuner.gp_tuner                    |
| PBTTuner        | tuners     | nni       | PBTTuner             | nni.pbt_tuner.pbt_tuner                  |
| SMAC            | tuners     | nni       | SMACTuner            | nni.smac_tuner.smac_tuner                |
| PPOTuner        | tuners     | nni       | PPOTuner             | nni.ppo_tuner.ppo_tuner                  |
| Medianstop      | assessors  | nni       | MedianstopAssessor   | nni.medianstop_assessor.medianstop_...   |
| Curvefitting    | assessors  | nni       | CurvefittingAssessor | nni.curvefitting_assessor.curvefitt...   |
| Hyperband       | advisors   | nni       | Hyperband            | nni.hyperband_advisor.hyperband_adv...   |
| BOHB            | advisors   | nni       | BOHB                 | nni.bohb_advisor.bohb_advisor            |
+-----------------+------------+-----------+----------------------+------------------------------------------+
Unregister builtin algorithms
Run following command to uninstall an installed package:

nnictl algo unregister <builtin name>

For example:
```bash
nnictl algo unregister demotuner
```