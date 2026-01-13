python -m data_synthesis.generate_scene --dataset replica --split train
python -m data_synthesis.generate_scene --dataset RoboTHOR --split train
python -m data_synthesis.generate_scene --dataset replica --split test



python -m data_synthesis.generate_scene --dataset ArchitecTHOR --split test
python -m data_synthesis.generate_scene --dataset iTHOR --split test
python -m data_synthesis.generate_scene --dataset RoboTHOR --split test


python -m data_synthesis.generate_scene --dataset ProcTHOR --split train
python -m data_synthesis.generate_scene --dataset ProcTHOR --split test

