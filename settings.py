from pathlib import Path


root = Path(__file__).parent                # setting.py的父目录
TRAINING_PATH = root / "outputs/training/"  # trainging results
EVAL_PATH = root / "outputs/validate/"        # evaluation results

Train_data_path = "/data/block0/hjw/Datasets/CoAlign+LightGlue/train_part/stage1_v4"
Val_data_path = "/data/block0/hjw/Datasets/CoAlign+LightGlue/val_part/stage1_v4"






