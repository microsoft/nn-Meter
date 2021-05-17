# DAGSplitter

Split pb models into kernels on given device

## Prerequisite

Please first use the tool `ruletest` provided by us to generate the rulefiles (or you can choose to handcraft the files), and replace `rulelib/rules` (default rulefiles are presented there).

## Installation

```
pip install -r requirements.txt
```

## Usage

Input models can be either json or pb. Please refer to `/data/raw.json` for json format.
To output readable results:
```
python main.py -i INPUT_MODELS [INPUT_MODELS ...] -f readable
```
