import numpy as np
from torch.utils.data import DataLoader
from dataset.utils import build_dataset
from trainers.get_trainer import get_trainer
from common.utils import save_exp_result


def cp(args):
    set_size_list = []
    coverage_list = []
    for run in range(args.num_runs):
        train_dataset, cal_dataset, test_dataset, num_classes = build_dataset(args)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        cal_loader = DataLoader(cal_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        trainer = get_trainer(args, num_classes)

        if args.epochs:
            trainer.train(train_loader, args.epochs)
        del train_loader
        del train_dataset

        trainer.predictor.calibrate(cal_loader)
        del cal_loader
        del cal_dataset

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
    train_dataset, _, test_dataset, num_classes = build_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    trainer = get_trainer(args, num_classes)

    trainer.train(train_loader, args.epochs)

    del train_loader
    del train_dataset


    result_dict = trainer.predictor.evaluate(test_loader)

    for key, value in result_dict.items():
        print(f'{key}: {value}')

    if args.save == "True":
        save_exp_result(args, result_dict)