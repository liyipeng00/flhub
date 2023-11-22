r"""Federated Averaging (FedAvg), Scheme II in "On the Convergence of FedAvg on Non-iid Data".

Sampling and Averaging Scheme:
\bar{x} = \sum_{s\in S} M/S p_s x_s,
where `S` clients is selected without replacement per round and `p_s=n_s/n` is the weight of client `s`. 

References:
https://github.com/lx10077/fedavgpy/
"""

import torch
import time
import copy
import numpy as np

from sim.algorithms.fedbase2 import FedClient, FedServer
from sim.data.data_utils import FedDataset
from sim.data.datasets import build_dataset
from sim.data.partition import build_partition
from sim.models.build_models import build_model
from sim.utils.record_utils import logconfig, add_log, record_exp_result
from sim.utils.utils import setup_seed, AverageMeter
from sim.utils.optim_utils import OptimKit, LrUpdater

import argparse
parser = argparse.ArgumentParser()
#models = ['mlp', 'lenet5', 'vgg11', 'resnet20']
parser.add_argument('-m', default='vgg11', type=str, help='Model')
parser.add_argument('-d', default='cifar10', type=str, help='Dataset')
parser.add_argument('-s', default=2, type=int, help='Index of split layer')
parser.add_argument('-R', default=200, type=int, help='Number of total training rounds')
parser.add_argument('-K', default=1, type=int, help='Number of local steps')
parser.add_argument('-M', default=100, type=int, help='Number of total clients')
parser.add_argument('-P', default=100, type=int, help='Number of clients participate')
parser.add_argument('--partition', default='dir', type=str, choices=['dir', 'iid', 'exdir'], help='Data partition')
parser.add_argument('--alpha', default=10, type=float, nargs='*', help='The parameter `alpha` of dirichlet distribution')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer')
parser.add_argument('--lr', default=0.0, type=float, help='Client/Local learning rate')
parser.add_argument('--lr-decay', default=1.0, type=float, help='Learning rate decay')
parser.add_argument('--momentum', default=0, type=float, help='Momentum of client optimizer')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay of client optimizer')
parser.add_argument('--batch-size', default=50, type=int, help='Mini-batch size')
parser.add_argument('--seed', default=1234, type=int, help='Seed')
parser.add_argument('--clip', default=0, type=int, help='Clip')
parser.add_argument('--log', default='', type=str, help='Log')
parser.add_argument('--device', default=0, type=int, help='Device')
args = parser.parse_args()

# nohup python main_fedavg2.py -m mlp -d mnist -s 1 -R 100 -K 10 -M 500 -P 10 --partition exdir --alpha 2 10 --optim sgd --lr 0.05 --lr-decay 0.9 --momentum 0 --batch-size 20 --seed 1234 --log Print &

class FedServer2(FedServer):
    def __init__(self):
        super().__init__()

    def select_clients(self, num_clients, num_clients_per_round):
        '''https://github.com/lx10077/fedavgpy/blob/master/src/trainers/base.py'''
        self.num_clients_per_round = min(num_clients_per_round, num_clients)
        self.num_clients = num_clients
        return np.random.choice(num_clients, num_clients_per_round, replace=False)

    def aggregate(self, local_params, num_all_instances):
        '''https://github.com/lx10077/fedavgpy/blob/master/src/trainers/fedavg4.py'''
        with torch.no_grad():
            param_avg = torch.zeros_like(local_params[0][0])
            for param, weight in local_params:
                param_avg.add_(weight * param)
            param_avg.div_(num_all_instances)
            return param_avg.mul_(self.num_clients / self.num_clients_per_round)

setup_seed(args.seed)
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
args.alpha = [int(args.alpha[0]), args.alpha[1]] if args.partition == 'exdir' else args.alpha

def customize_record_name(args):
    '''FedAvg2_M10_P10_K2_R4_mlp_mnist_exdir2,10.0_sgd0.001,1.0,0.0,0.0001_b20_seed1234_clip0.csv'''
    record_name = f'FedAvg2_M{args.M}_P{args.P}_K{args.K}_R{args.R}_{args.m}_{args.d}_{args.partition}{args.alpha[0]},{args.alpha[1]}'\
                + f'_{args.optim}{args.lr},{args.lr_decay},{args.momentum},{args.weight_decay}_b{args.batch_size}_seed{args.seed}_clip{args.clip}'
    return record_name
record_name = customize_record_name(args)

def main():
    global args, record_name, device
    logconfig(name=record_name, flag=args.log)
    add_log('record_name: {}'.format(record_name), flag=args.log)
    
    client = FedClient()
    server = FedServer2()

    train_dataset, test_dataset = build_dataset(args.d)
    net_dataidx_map = build_partition(dataset=args.d, n_nets=args.M, partition=args.partition, alpha=[args.alpha[0], args.alpha[1]])
    train_feddataset = FedDataset(train_dataset, net_dataidx_map)
    client.setup_train_dataset(train_feddataset)
    client.setup_test_dataset(test_dataset)

    global_model = build_model(model=args.m, dataset=args.d)
    server.setup_model(global_model.to(device))

    # construct optim kit
    client_optim_kit = OptimKit(optim_name=args.optim, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    client_optim_kit.setup_lr_updater(LrUpdater.exponential_lr_updater, mul=args.lr_decay)
    client.setup_optim_kit(client_optim_kit)
    client.setup_criterion(torch.nn.CrossEntropyLoss())

    start_time = time.time()
    
    for round in range(args.R):
        selected_clients = server.select_clients(args.M, args.P)
        add_log('selected clients: {}'.format(selected_clients), flag=args.log)
        local_params = []
        for c_id in selected_clients:
            client_local_param = client.local_update_step(model=copy.deepcopy(server.global_model), dataset=client.train_feddataset.get_dataset(c_id), num_steps=args.K, device=device, clip=args.clip)
            local_params.append([client_local_param, client.train_feddataset.get_datasetsize(c_id)])
        param_avg = server.aggregate(local_params, client.train_feddataset.totalsize)
        torch.nn.utils.vector_to_parameters(param_avg, server.global_model.parameters())
        
        client.optim_kit.update_lr()
        #add_log('lr={}'.format(client.optim_kit.settings['lr']), flag=args.log)

        # evaluate on train dataset
        train_losses, train_top1, train_top5 = AverageMeter(), AverageMeter(), AverageMeter()
        for c_id in selected_clients:
            local_losses, local_top1, local_top5 = \
            client.evaluate_dataset(model=server.global_model, dataset=client.train_feddataset.get_dataset(c_id), device=args.device)
            train_losses.update(local_losses.avg, local_losses.count), train_top1.update(local_top1.avg, local_top1.count), train_top5.update(local_top5.avg, local_top5.count)
        add_log("Round {}'s server train acc: {:.2f}%, train loss: {:.4f}".format(round+1, train_top1.avg, train_losses.avg), 'green', flag=args.log)
        
        # evaluate on test dataset
        test_losses, test_top1, test_top5 = client.evaluate_dataset(model=server.global_model, dataset=client.test_dataset, device=args.device)
        add_log("Round {}'s server test  acc: {:.2f}%, test  loss: {:.4f}".format(round+1, test_top1.avg, test_losses.avg), 'red', flag=args.log)

        record_exp_result(record_name, 
                          {'round':round+1, 'train_loss':train_losses.avg, 'test_loss':test_losses.avg, 'train_top1':train_top1.avg, 'test_top1':test_top1.avg, 'train_top5':train_top5.avg, 'test_top5':test_top5.avg})
    end_time = time.time()
    add_log("TrainingTime: {} sec".format(end_time - start_time), flag=args.log)

if __name__ == '__main__':
    main()