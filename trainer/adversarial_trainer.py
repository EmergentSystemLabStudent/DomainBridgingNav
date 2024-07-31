
from __future__ import annotations

import sys
from typing import Tuple
from pathlib import Path 
from tqdm import tqdm

import numpy as np 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import adjusted_rand_score as ARI

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from model.supervised_simsiam import SimSiam
from .modules.schedulars import CosineScheduler
from utils.index_colormap import IndexColorMap

class Trainer:
    
    _model: SimSiam
    _dataloader: DataLoader
    _val_dataloader: DataLoader
    _eval_loader: DataLoader
    _test_loader: DataLoader
    _loss: nn.CosineSimilarity
    _scheduler: CosineScheduler
    _writer: SummaryWriter
    _index_color_map: IndexColorMap
    
    def __init__(
        self, 
        model: SimSiam,
        dataloader: DataLoader,
        val_dataloader: DataLoader, 
        eval_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int=30,
        init_lr: float=0.03,
        weight_decay: float=1.5e-6, 
        log_dir: str="./logs",
        checkpoint_dir: str="./checkpoints",
        device: str="cuda", 
        name: str="base_simsiam"
    )->None:
        
        ## --- pytorch modules --- ##
        self._model = model 
        self._cosine_sim = nn.CosineSimilarity(dim=1)
        self._cross_entropy = nn.CrossEntropyLoss()
        self._dataloader = dataloader
        self._val_dataloader = val_dataloader
        self._eval_loader = eval_loader
        self._test_loader = test_loader
        self._optimizer = SGD(self._model.parameters(), 
                              lr=init_lr, 
                              weight_decay=weight_decay, 
                              momentum=0.9)
        self._scheduler = CosineScheduler()
        self._lambda_ce = 0.1
        
        ## --- other attributes --- ##
        self._writer = SummaryWriter(log_dir=log_dir)
        self._index_color_map = IndexColorMap(n=len(dataloader.dataset))
        self._epochs = num_epochs
        self._init_lr = init_lr
        self._device = device
        self._checkpoint_dir = checkpoint_dir
        self._name = name
        
    def run(self, autocast: bool=True) -> None:
        
        self._model = self._model.to(self._device)
        
        if autocast:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in tqdm(range(self._epochs)):
            self._model.train()
            self._scheduler.adsust_lr(self._optimizer, epoch, self._epochs, self._init_lr)
            loss_log = []
            cosine_sim_log = []
            cross_entropy_log = []
            domain_discrimination_loss_log = []
            
            eval_ari, test_ari, all_ari = self._all_ari()
            self._writer.add_scalar("ari/eval_dataset", eval_ari, epoch)
            self._writer.add_scalar("ari/test_dataset", test_ari, epoch)
            self._writer.add_scalar("ari/cross_domain", all_ari, epoch)
            self._model.train()
            for minibatch in tqdm(self._dataloader):
                view_batch1, view_batch2, instance_labels, domain_labels1, domain_labels2 = minibatch[0].to(self._device), minibatch[1].to(self._device), minibatch[2].to(self._device), minibatch[3].to(self._device), minibatch[4].to(self._device)
                self._optimizer.zero_grad()
                
                if autocast:
                    with torch.cuda.amp.autocast():
                        p1, p2, z1, z2, logits1, logits2, domain_logits1, domain_logits2 = self._model(view_batch1, view_batch2)
                        
                        cosine_loss = -(self._cosine_sim(p1, z2).mean() + self._cosine_sim(p2, z1).mean())*0.5
                        classfier_loss = (self._cross_entropy(logits1, instance_labels) + self._cross_entropy(logits2, instance_labels))*0.5
                        domain_discrimination_loss = (self._cross_entropy(domain_logits1, domain_labels1)+self._cross_entropy(domain_logits2, domain_labels2))*0.5
                        loss = cosine_loss + classfier_loss + domain_discrimination_loss
                        
                        scaler.scale(loss).backward()
                        scaler.step(self._optimizer)
                        scaler.update()
                        
                else:
                    p1, p2, z1, z2, logits1, logits2, domain_logits1, domain_logits2 = self._model(view_batch1, view_batch2)
                    
                    cosine_loss = -(self._cosine_sim(p1, z2).mean() + self._cosine_sim(p2, z1).mean())*0.5
                    classfier_loss = (self._cross_entropy(logits1, instance_labels) + self._cross_entropy(logits2, instance_labels))*0.5
                    domain_discrimination_loss = (self._cross_entropy(domain_logits1, domain_labels1)+self._cross_entropy(domain_logits2, domain_labels2))*0.5
                    loss = cosine_loss + classfier_loss + domain_discrimination_loss
                    loss.backward()
                    self._optimizer.step()
                    
                loss_log.append(loss.item())
                cosine_sim_log.append(cosine_loss.item())
                cross_entropy_log.append(classfier_loss.item())
                domain_discrimination_loss_log.append(domain_discrimination_loss.item())
                
            self._writer.add_scalar("loss/train", sum(loss_log)/len(loss_log), epoch)
            self._writer.add_scalar("cosine_sim/train", sum(cosine_sim_log)/len(cosine_sim_log), epoch)
            self._writer.add_scalar("cross_entropy/train", sum(cross_entropy_log)/len(cross_entropy_log), epoch)
            self._writer.add_scalar("domain_discrimination/train", sum(domain_discrimination_loss_log)/len(domain_discrimination_loss_log), epoch)
            
            validation_loss, validation_cosine_sim, validation_cross_entropy, validation_domain_discrimination = self._validation_loss(autocast)
            self._writer.add_scalar("loss/validation", validation_loss, epoch)
            self._writer.add_scalar("cosine_sim/validation", validation_cosine_sim, epoch)
            self._writer.add_scalar("cross_entropy/validation", validation_cross_entropy, epoch)
            self._writer.add_scalar("domain_discrimination/validation", validation_domain_discrimination, epoch)
            
            # self._visuzlize_latent_space(self._eval_loader, epoch, self._name)
            
            # --- Save Weight --- #
            if (epoch + 1) % 100 == 0 :
                state = {"epoch": epoch + 1, "state_dict": self._model.state_dict(), "optimizer": self._optimizer.state_dict()}
                self._save_checkpoint(state, self._checkpoint_dir + f"/{epoch + 1}.pth.tar")
        
        eval_ari, test_ari, all_ari = self._all_ari()
        self._writer.add_scalar("ari/eval_dataset", eval_ari, epoch)
        self._writer.add_scalar("ari/test_dataset", test_ari, epoch)
        self._writer.add_scalar("ari/cross_domain", all_ari, epoch)  
    
        state = {"epoch": epoch + 1, "state_dict": self._model.state_dict(), "optimizer": self._optimizer.state_dict()}
        self._save_checkpoint(state, self._checkpoint_dir + f"/{epoch + 1}.pth.tar")
        
    def _validation_loss(self, autocast: bool=True) -> Tuple[float, float, float, float]:
        
        loss_log = []
        cosine_sim_log = []
        cross_entropy_log = []
        domain_discrimination_loss_log = []
        
        if autocast:
            scaler = torch.cuda.amp.GradScaler()
            
        with torch.no_grad():
            for minibatch in tqdm(self._val_dataloader):
                view_batch1, view_batch2, instance_labels, domain_labels1, domain_labels2 = minibatch[0].to(self._device), minibatch[1].to(self._device), minibatch[2].to(self._device), minibatch[3].to(self._device), minibatch[4].to(self._device)
                self._optimizer.zero_grad()
                
                if autocast:
                    with torch.cuda.amp.autocast():
                        p1, p2, z1, z2, logits1, logits2, domain_logits1, domain_logits2 = self._model(view_batch1, view_batch2)
                    
                    cosine_loss = -(self._cosine_sim(p1, z2).mean() + self._cosine_sim(p2, z1).mean())*0.5
                    classfier_loss = (self._cross_entropy(logits1, instance_labels) + self._cross_entropy(logits2, instance_labels))*0.5
                    domain_discrimination_loss = (self._cross_entropy(domain_logits1, domain_labels1)+self._cross_entropy(domain_logits2, domain_labels2))*0.5
                    loss = cosine_loss + classfier_loss + domain_discrimination_loss
                    
                else:
                    p1, p2, z1, z2, logits1, logits2, domain_logits1, domain_logits2 = self._model(view_batch1, view_batch2)
                    
                    cosine_loss = -(self._cosine_sim(p1, z2).mean() + self._cosine_sim(p2, z1).mean())*0.5
                    classfier_loss = (self._cross_entropy(logits1, instance_labels) + self._cross_entropy(logits2, instance_labels))*0.5
                    domain_discrimination_loss = (self._cross_entropy(domain_logits1, domain_labels1)+self._cross_entropy(domain_logits2, domain_labels2))*0.5
                    loss = cosine_loss + classfier_loss + domain_discrimination_loss
                        

                loss_log.append(loss.item())
                cosine_sim_log.append(cosine_loss.item())
                cross_entropy_log.append(classfier_loss.item())
                domain_discrimination_loss_log.append(domain_discrimination_loss.item())
                
        return sum(loss_log)/len(loss_log), sum(cosine_sim_log)/len(cosine_sim_log), sum(cross_entropy_log)/len(cross_entropy_log), sum(domain_discrimination_loss_log)/len(domain_discrimination_loss_log)
    
    def _all_ari(self) -> Tuple[float, float, float]:
        
        robot_embedding_list = []
        robot_label_list = []
        self._model.eval()
        for images, labels in tqdm(self._eval_loader):
            embeddings = self._model.get_embedding(images.to(self._device))
            robot_embedding_list += embeddings.tolist()
            robot_label_list += labels.detach().cpu().numpy().tolist()
        robot_embedding_list = np.array(robot_embedding_list)
        robot_label_list = np.array(robot_label_list)
        
        k_means = KMeans(n_clusters=len(self._eval_loader.dataset.classes), init = 'random', max_iter = 100, n_init = 10,  random_state=0).fit(normalize(robot_embedding_list))
        robot_k_means_cluster = k_means.labels_
        eval_ari_score = ARI(robot_label_list, robot_k_means_cluster)
        
        query_embedding_list = []
        query_label_list = []
        for images, labels in tqdm(self._test_loader):
            embeddings = self._model.get_embedding(images.to(self._device))
            query_embedding_list += embeddings.tolist()
            query_label_list += labels.detach().cpu().numpy().tolist()
            
        query_embedding_list = np.array(query_embedding_list)
        query_label_list = np.array(query_label_list)
        
        k_means = KMeans(n_clusters=len(self._test_loader.dataset.classes), init = 'random', max_iter = 100, n_init = 10,  random_state=0).fit(normalize(query_embedding_list))
        query_k_means_cluster = k_means.labels_
        query_ari_score = ARI(query_label_list, query_k_means_cluster)
        
        all_embedding_list = robot_embedding_list.tolist() + query_embedding_list.tolist()
        all_label_list = robot_label_list.tolist() + query_label_list.tolist()
        k_means = KMeans(n_clusters=len(self._test_loader.dataset.classes), init = 'random', max_iter = 100, n_init = 10,  random_state=0).fit(normalize(all_embedding_list))
        all_k_means_cluster = k_means.labels_
        cross_domain_ari_score = ARI(all_label_list, all_k_means_cluster)
        
        return eval_ari_score, query_ari_score, cross_domain_ari_score
    
    def _visuzlize_latent_space(self, eval_dataloader, epoch: int, name: str) -> None:
        
        label_log = []
        embedding_log = []
        self._model.eval()
        for images, labels in eval_dataloader:
            embeddings = self._model.get_embedding(images.to(self._device))
            label_log += labels.detach().cpu().numpy().tolist()
            embedding_log += embeddings.tolist()
            
        label_log = np.array(label_log)
        embedding_log = np.array(embedding_log)
        
        save_folder = Path(__file__).parent.parent.absolute() / "result" / name
        if not save_folder.exists():
            save_folder.mkdir(parents=True)
        
        ## TSNE ##
        tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
        decomposed_embedding = tsne.fit_transform(embedding_log)
        figure = plt.figure()
        for label in np.unique(label_log):
            color = self._index_color_map.get_color(label)
            indicies = np.where(label_log == label)
            targets = decomposed_embedding[indicies]
            plt.scatter(targets[:, 0], targets[:, 1], marker=f"${label}$", alpha=0.3, color=color)
        
        plt.savefig(save_folder / (str(epoch) + "_tsne.png"))
        
        ## PCA ##
        pca = PCA(n_components=2)
        pca.fit(embedding_log)
        decomposed_embedding = pca.transform(embedding_log)
        figure = plt.figure()
        for label in np.unique(label_log):
            color = self._index_color_map.get_color(label)
            indicies = np.where(label_log == label)
            targets = decomposed_embedding[indicies]
            plt.scatter(targets[:, 0], targets[:, 1], marker=f"${label}$", alpha=0.3, color=color)
        
        plt.savefig(save_folder / (str(epoch) + "_pca.png"))
        
    def _save_checkpoint(self, state, filename) -> None:
        torch.save(state, filename)