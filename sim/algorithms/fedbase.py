import numpy as np
import torch
import copy
from torch.utils.data import DataLoader
from sim.utils.utils import AverageMeter, accuracy


eval_batch_size = 32

###### CLIENT ######
class FedClient():
    def __init__(self):
        pass
    
    def setup_criterion(self, criterion):
        self.criterion = criterion

    def setup_feddataset(self, dataset):
        self.feddataset = dataset

    def setup_optim_kit(self, optim_kit):
        self.optim_kit = optim_kit
    
    #client.local_update_step(model=copy.deepcopy(server.global_model), dataset=client.train_feddataset.get_dataset(c_id), num_steps=args.K, device=device, clip=args.clip)
    def local_update_step(self, c_id, model, num_steps, device, **kwargs):
        dataset=self.feddataset.get_dataset(c_id)
        data_loader = DataLoader(dataset, batch_size=self.optim_kit.batch_size, shuffle=True)
        optimizer = self.optim_kit.optim(model.parameters(), **self.optim_kit.settings)

        prev_model = copy.deepcopy(model)
        model.train()
        step_count = 0
        while(True):
            for input, target in data_loader:
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = self.criterion(output, target)
                optimizer.zero_grad()
                loss.backward()

                if 'clip' in kwargs.keys() and kwargs['clip'] > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=kwargs['clip'])

                optimizer.step()
                step_count += 1
                if (step_count >= num_steps):
                    break
            if (step_count >= num_steps):
                break
        with torch.no_grad():
            curr_vec = torch.nn.utils.parameters_to_vector(model.parameters())
            prev_vec = torch.nn.utils.parameters_to_vector(prev_model.parameters())
            delta_vec = curr_vec - prev_vec
            assert step_count == num_steps            
            # add log
            local_log = {}
            local_log = {'total_norm': total_norm} if 'clip' in kwargs.keys() and kwargs['clip'] > 0 else local_log
            return delta_vec, local_log

    def local_update_epoch(self, client_model,data, epoch, batchsize):
        pass

    def evaluate_dataset(self, model, dataset, device):
        '''Evaluate on the given dataset'''
        data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for input, target in data_loader:
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = self.criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=[1,5])
                losses.update(loss.item(), target.size(0))
                top1.update(acc1.item(), target.size(0))
                top5.update(acc5.item(), target.size(0))

            return losses, top1, top5,

###### SERVER ######
class FedServer():
    def __init__(self):
        super(FedServer, self).__init__()
    
    def setup_model(self, model):
        self.global_model = model
    
    def setup_optim_settings(self, **settings):
        self.lr = settings['lr']
    
    def select_clients(self, num_clients, num_clients_per_round):
        '''https://github.com/lx10077/fedavgpy/blob/master/src/trainers/base.py'''
        num_clients_per_round = min(num_clients_per_round, num_clients)
        return np.random.choice(num_clients, num_clients_per_round, replace=False)
    
    def global_update(self):
        with torch.no_grad():
            param_vec_curr = torch.nn.utils.parameters_to_vector(self.global_model.parameters()) + self.lr * self.delta_avg 
            return param_vec_curr
    
    def aggregate_reset(self):
        self.delta_avg = None
        self.weight_sum = torch.tensor(0) 
    
    def aggregate_update(self, local_delta, weight):
        with torch.no_grad():
            if self.delta_avg == None:
                self.delta_avg = torch.zeros_like(local_delta)
            self.delta_avg.add_(weight * local_delta)
            self.weight_sum.add_(weight)
    
    def aggregate_avg(self):
        with torch.no_grad():
            self.delta_avg.div_(self.weight_sum)

    