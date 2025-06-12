import numpy as np
from torch.utils.data import DataLoader
from dataset.utils import build_train_dataloader, build_cal_test_loader
from trainers.get_trainer import get_trainer
from common.utils import save_exp_result


def cp(args):
    set_size_list = []
    coverage_list = []
    for run in range(args.num_runs):
        cal_loader, test_loader = build_cal_test_loader(args)

        trainer = get_trainer(args)

        if args.epochs and args.epochs > 0:
            train_dataloader = build_train_dataloader(args)
            trainer.train(train_dataloader)

        trainer.predictor.calibrate(cal_loader)

        result_dict = trainer.predictor.evaluate(test_loader)

        for key, value in result_dict.items():
            print(f'{key}: {value}')
        coverage_list.append(result_dict["Coverage"])
        set_size_list.append(result_dict["AverageSetSize"])

        if args.save == "True":
            save_exp_result(args, result_dict)
    print()
    print("Mean result")
    mean_coverage = np.array(coverage_list).mean()
    mean_set_size = np.array(set_size_list).mean()
    print(f"Mean Coverage: {mean_coverage}")
    print(f"Mean Set Size: {mean_set_size}")


def standard(args):


    train_loader = build_train_dataloader(args)

    trainer = get_trainer(args)

    trainer.train(train_loader, args.epochs)

    del train_loader

    _, test_loader = build_cal_test_loader(args)
    result_dict = trainer.predictor.evaluate(test_loader)

    for key, value in result_dict.items():
        print(f'{key}: {value}')

    if args.save == "True":
        save_exp_result(args, result_dict)