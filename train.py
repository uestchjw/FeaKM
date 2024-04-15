
from pathlib import Path
import logging
import math
import numpy as np
import argparse
import importlib
from collections import defaultdict
from tqdm import tqdm
from pydoc import locate



from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast



from utils.setup_logging import setup_logging
from utils.tensor import batch_to_device
from utils.tools import (
    AverageMetric,
    MedianMetric,
    PRMetric,
    RecallMetric,
    fork_rng,
    set_seed,
)
from utils.experiments import get_best_checkpoint, get_last_checkpoint, save_experiment



def pack_lr_parameters(params, base_lr, lr_scaling):
    """Pack each group of parameters with the respective scaled learning rate."""
    filters, scales = tuple(zip(*[(n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logging.info(
        "Parameters with scaled learning rate:\n%s",
        {s: [n for n, _ in ps] for s, ps in scale2params.items() if s != 1},
    )
    lr_params = [
        {"lr": scale * base_lr, "params": [p for _, p in ps]}
        for scale, ps in scale2params.items()
    ]
    return lr_params

def get_lr_scheduler(optimizer, conf):
    """Get lr scheduler specified by conf.train.lr_schedule."""
    if conf.type not in ["factor", "exp", None]:
        return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, **conf.options)

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        if conf.type is None:
            return 1
        if conf.type == "factor":
            return 1.0 if it < conf.start else conf.factor
        if conf.type == "exp":
            gam = 10 ** (-1 / conf.exp_div_10)
            return 1.0 if it < conf.start else gam
        else:
            raise ValueError(conf.type)

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)

@torch.no_grad()
def do_evaluation(model, loader, device, loss_fn, conf, pbar=True):
    model.eval()
    results = {}
    pr_metrics = defaultdict(PRMetric)
    figures = []
    if conf.plot is not None:
        n, plot_fn = conf.plot
        plot_ids = np.random.choice(len(loader), min(len(loader), n), replace=False)
    #! origin:
    # for i, data in enumerate(
    #     tqdm(loader, desc="Evaluation", ascii=True, disable=not pbar)
    # ):
    for i, data in enumerate(loader):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses, metrics = loss_fn(pred, data)
            if conf.plot is not None and i in plot_ids:
                figures.append(locate(plot_fn)(pred, data))
            # add PR curves
            for k, v in conf.pr_curves.items():
                pr_metrics[k].update(
                    pred[v["labels"]],
                    pred[v["predictions"]],
                    mask=pred[v["mask"]] if "mask" in v.keys() else None,
                )
            del pred, data
        numbers = {**metrics, **{"loss/" + k: v for k, v in losses.items()}}
        # numbers.keys(): ['match_recall', 'match_precision', 'accuracy', 'average_precision', 'loss/total', 'loss/last', 'loss/assignment_nll', 'loss/nll_pos', 'loss/nll_neg', \
                        # 'loss/num_matchable', 'loss/num_unmatchable', 'loss/row_norm']

        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
                if k in conf.median_metrics:
                    results[k + "_median"] = MedianMetric()
                if k in conf.recall_metrics.keys():
                    q = conf.recall_metrics[k]
                    results[k + f"_recall{int(q)}"] = RecallMetric(q)
            results[k].update(v)
            if k in conf.median_metrics:
                results[k + "_median"].update(v)
            if k in conf.recall_metrics.keys():
                q = conf.recall_metrics[k]
                results[k + f"_recall{int(q)}"].update(v)
        del numbers
    results = {k: results[k].compute() for k in results}
    # print(results["loss/num_matchable"]) # 6.055338177752935
    # raise Exception
    return results, {k: v.compute() for k, v in pr_metrics.items()}, figures




default_train_conf = {
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "lr_schedule": {
        "type": None,  # string in {factor, exp, member of torch.optim.lr_scheduler}
        "start": 0,
        "exp_div_10": 0,
        "on_epoch": False,
        "factor": 1.0,
        "options": {},  # add lr_scheduler arguments here
    },
    "lr_scaling": [(100, ["dampingnet.const"])],
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "log_grad_every_iter": None,  # interval for logging gradient hists
    "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
    "keep_last_checkpoints": 5,  # keep only the last X checkpoints
    "load_experiment": None,  # initialize the model from a previous experiment
    "median_metrics": [],  # add the median of some metrics
    "recall_metrics": {},  # add the recall of some metrics
    "pr_metrics": {},  # add pr curves, set labels/predictions/mask keys
    "best_key": "loss/total",  # key to use to select the best checkpoint
    "dataset_callback_fn": None,  # data func called at the start of each epoch
    "dataset_callback_on_val": False,  # call data func on val data?
    "clip_grad": None,
    "pr_curves": {},
    "plot": None,
    "submodules": [],
}
default_train_conf = OmegaConf.create(default_train_conf)



def training(conf, output_dir, args):
    if conf.restore:
        pass
    else:
        epoch = 0
    best_eval = float("inf") #! best_eval是一个scalar
    conf.train = OmegaConf.merge(default_train_conf, conf.train)

    #! logging and tensorboard
    log_filename = Path(output_dir, 'logging.txt')
    setup_logging(log_filename)
    logging.info(f"Starting experiment: {conf.experiment}")
    writer = SummaryWriter(log_dir=str(output_dir))
    
    set_seed(conf.train.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device {device}")
    
    #! dataset and dataloader
    mod = importlib.import_module("datasets." + conf.data.name) 
    train_dataset = getattr(mod,conf.data.name)("train")
    val_dataset = getattr(mod,conf.data.name)("val")
    train_loader = DataLoader(train_dataset, 
                              batch_size=conf.train.batch_size,
                              shuffle=conf.train.shuffle,
                              num_workers=conf.train.num_workers,
                              drop_last=conf.train.drop_last)

    val_loader = DataLoader(val_dataset, 
                              batch_size=conf.val.batch_size,
                              shuffle=conf.val.shuffle,
                              num_workers=conf.val.num_workers,
                              drop_last=conf.val.drop_last)
    logging.info(f"Training loader has {len(train_loader)} batches")
    logging.info(f"Validation loader has {len(val_loader)} batches")
    
    
    #! model
    mod = importlib.import_module("models." + conf.model.name)
    model = getattr(mod, conf.model.name)(conf).to(device)

    logging.info(f"Model: {conf.model.name}\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total_params: {total_params}') # 12498673

    #! loss function
    loss_fn = model.loss
    
    #! optimizer
    optimizer_fn = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    }[conf.train.optimizer] #! adam
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    lr_params = pack_lr_parameters(params, conf.train.lr, conf.train.lr_scaling)
    optimizer = optimizer_fn(
        lr_params, lr=conf.train.lr, **conf.train.optimizer_options
    )
    #! scaler
    scaler = GradScaler(enabled=args.mixed_precision is not None)
    logging.info(f"Training with mixed precision: {args.mixed_precision}")
    
    mp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        None: torch.float32, # 默认为float32
    }[args.mixed_precision]
    
    #! lr_scheduler
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, conf=conf.train.lr_schedule)
    
    
    logging.info("Starting training with configuration:\n%s", OmegaConf.to_yaml(conf))
    rank = 0
    losses_ = None
    stop = False
    #! start training
    while epoch < conf.train.epoches:
        logging.info(f"Starting epoch: {epoch}")

        # todo: 这里为什么每个epoch的seed都要不同？是因为前面设置了相同的全局seed吗？
        # set the seed
        set_seed(conf.train.seed + epoch)
        
        # update learning rate
        if conf.train.lr_schedule.on_epoch and epoch > 0: #! true
            old_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
            logging.info(f'lr changed from {old_lr} to {optimizer.param_groups[0]["lr"]}')
        
        model.train()
        for it, data in enumerate(train_loader):
            # tot_it = (len(train_loader) * epoch + it)
            # tot_n_samples = tot_it
            # if not args.log_it:
            #     # We normalize the x-axis of tensorflow to num samples!
            #     tot_n_samples *= train_loader.batch_size

            optimizer.zero_grad()
            with autocast(enabled=args.mixed_precision is not None, dtype=mp_dtype):
                data = batch_to_device(data, device, non_blocking=True)
                pred = model(data) 
                losses, _ = loss_fn(pred, data)    
                loss = torch.mean(losses["total"])
            
            # todo: 删除为nan的loss, 以前没见过这种操作
            if torch.isnan(loss).any():
                print(f'Detected NaN, skipping iteration {it}')
                del pred, data, loss, losses
                continue
            
            # todo: 判断loss是否需要反向传播, 以前没见过这种操作
            # todo: 这种情况什么时候发生？
            do_backward = loss.requires_grad

            if do_backward:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: 
                logging.warning(f"Skip iteration {it} due to detach.")
            # losses.keys(): ['total', 'last', 'assignment_nll', 'nll_pos', 'nll_neg', 'num_matchable', 'num_unmatchable', 'confidence', 'row_norm']
            # losses["num_matchable"]: [7., 7., 1., 9., 3., 1., 2., 6., 6., 4., 8., 1., 5., 6., 3., 4.]

        #! 这里的losses, 永远是最后一个batch的, 即[b, ]

        # lossses.keys(): ['total', 'last', 'assignment_nll', 'nll_pos', 'nll_neg', 'num_matchable', 'num_unmatchable', 'confidence', 'row_norm']
        #! log training loss: 每个epoch做一次
        for k in sorted(losses.keys()):
            losses[k] = torch.mean(losses[k], -1)
            losses[k] = losses[k].item()
        # losses["num_matchable"]: 4.375 
        
        str_losses = [f"{k} {v:.3E}" for k, v in losses.items()]
        # ['total 1.612E+00', 'last 1.500E+00', 'assignment_nll 1.500E+00', 'nll_pos 2.833E+00', 'nll_neg 1.679E-01', \
        #'num_matchable 4.375E+00', 'num_unmatchable 1.234E+02', 'confidence 8.349E-02', 'row_norm 9.304E-01']
        logging.info( "[E {} | it {}] loss {{{}}}".format( epoch, it, ", ".join(str_losses) ) ) #! 这里之前copy错了, it {} 没复制过来
        for k, v in losses.items():
            writer.add_scalar("training/" + k, v, epoch)
        writer.add_scalar("training/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("training/epoch", epoch, epoch)
        del pred, data, loss, losses

        #! v2和v1的区别: 不是每个epoch才做一次验证
        #! v3: 每个epoch进行一次验证
        
        # Run validation
        with fork_rng(seed=conf.train.seed):
            results, pr_metrics, figures = do_evaluation(
                model,
                val_loader,
                device,
                loss_fn,
                conf.train,
                pbar=(rank == -1))

        if rank == 0:
            str_results = [ f"{k} {v:.3E}" for k, v in results.items() if isinstance(v, float)]
            logging.info(f'[Validation] {{{", ".join(str_results)}}}')
            for k, v in results.items():
                if isinstance(v, dict):
                    writer.add_scalars(f"figure/val/{k}", v, epoch)
                else:
                    writer.add_scalar("val/" + k, v, epoch)
            for k, v in pr_metrics.items():
                writer.add_pr_curve("val/" + k, *v, epoch)
            # @TODO: optional always save checkpoint
            if results[conf.train.best_key] < best_eval:
                best_eval = results[conf.train.best_key]
                save_experiment(
                    model,
                    optimizer,
                    lr_scheduler,
                    conf,
                    losses_,
                    results,
                    best_eval,
                    epoch,
                    "",
                    output_dir,
                    stop,
                    args.distributed,
                    cp_name="checkpoint_best.tar",
                )
                logging.info(f"New best val: {conf.train.best_key}={best_eval}")
            if len(figures) > 0:
                for i, figs in enumerate(figures):
                    for name, fig in figs.items():
                        writer.add_figure(f"figures/{i}_{name}", fig, epoch)
        torch.cuda.empty_cache()  # should be cleared at the first iter

        #! save model
        if rank == 0:
            best_eval = save_experiment(
                model,
                optimizer,
                lr_scheduler,
                conf,
                losses_,
                results,
                best_eval,
                epoch,
                "",
                output_dir=output_dir,
                stop=stop,
                distributed=args.distributed,
            )
        
        epoch += 1
    
    logging.info("Finished training")
    writer.close()  #* 只有关闭writer, 才会把缓存的数据写到events文件中




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str)
    parser.add_argument("--conf_path", type=str)
    
    parser.add_argument("--mixed_precision", "--mp", default = None)
    
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--log_it", action="store_true")
    parser.add_argument("--no_eval_0", action="store_true")
    parser.add_argument("--run_benchmarks", action="store_true")
    args = parser.parse_args()

    from settings import TRAINING_PATH
    output_dir = Path(TRAINING_PATH, args.experiment)
    output_dir.mkdir(parents=True, exist_ok=True) # 递归创建实验目录
    
    conf = OmegaConf.load(args.conf_path)
    training(conf, output_dir, args)
