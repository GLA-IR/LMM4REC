# coding: utf-8
# @email  : enoche.chow@gmail.com

import os
import numpy as np
import torch
import torch.nn as nn
from models import clip
from utils.dataset import create_dataset, create_loader
from models.vlmo.vlmo_module import VLMo

class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError
    #
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # load encoded features here
        self.v_feat, self.t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            # if file exist?
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)

            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'


class MultiModalEndtoEndRecommender(AbstractRecommender):
    def __init__(self, config, dataloader):
        super(MultiModalEndtoEndRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']
        self.config = config
        # build encoder
        self.multi_modal_encoder = self.build_multiModalEncoder(config)

        self.multimodal_data_loader = self.build_dataloader(config)

        # build embeddings
        self.v_feat, self.t_feat = self.build_embeds(config,
                                                     self.multi_modal_encoder,
                                                     self.multimodal_data_loader)


    def build_dataloader(self, config):
        # creat dataset
        datasets = [create_dataset('movie_dataset', config)]
        samplers = [None]
        # create dataloader
        data_loader = \
            create_loader(datasets, samplers,
                          batch_size=[16],
                          num_workers=[2], is_trains=[False],
                          collate_fns=[None])[0]
        return data_loader

    def build_multiModalEncoder(self, config):
        # build vision encoder
        if 'vlmo' in config['multi_modal_encoder']:
            print('Using vlmo as multi encoder')
            multimodal_encoder = VLMo(config)

            if config['checkpoint'] is not None:
                # msg = multimodal_encoder.load_state_dict(torch.load(config['checkpoint']))
                # print(msg)
                print('load checkpoint from {}'.format(config['checkpoint']))

            multimodal_encoder = multimodal_encoder.to(config['device'])

        elif 'clip' in config['multi_modal_encoder']:
            print('Using clip as multi encoder')
            multimodal_encoder, self.clip_preprocess = clip.load("ViT-B/16")

        return multimodal_encoder


    def build_embeds(self, config, multimodal_encoder, data_loader):
        if 'clip' in config['multi_modal_encoder']:
            text_embeddings = []
            image_embeddings = []
            for batch in data_loader:
                images = batch['image'][0]
                images = images.cuda()
                texts = batch['text']
                image_input = images  # torch.tensor(np.stack(images)).cuda()
                text_tokens = clip.tokenize([desc[0:77] for desc in texts]).cuda()

                with torch.no_grad():
                    image_features = multimodal_encoder.encode_image(image_input).float()
                    text_features = multimodal_encoder.encode_text(text_tokens).float()

                # with vl layers
                for embed in image_features.unbind():
                    image_embeddings.append(embed.detach().cpu().numpy())
                for embed in text_features:
                    text_embeddings.append(embed.detach().cpu().numpy())

            return torch.FloatTensor(image_embeddings).to(
                    self.device), torch.FloatTensor(text_embeddings).to(
                    self.device)

        elif 'vlmo' in config['multi_modal_encoder']:
            text_embeddings = []
            image_embeddings = []
            cls_features = []
            raw_cls_features = []
            text_embeddings_infer_text = []
            text_embeddings_infer_text_vl = []
            text_embeddings_infer_text_ft = []
            image_embeddings_infer_image = []
            image_embeddings_infer_image_vl = []
            image_embeddings_infer_image_ft = []

            for batch in data_loader:

                # with vl layers
                # ret_dict = model.infer_text(batch, mask_text=False)
                # for embed in ret_dict['text_feats'].unbind():
                #     text_embeddings_infer_text.append(embed[0].detach().cpu().numpy())
                #
                # for embed in ret_dict['cls_vlffn_feats'].unbind():
                #     text_embeddings_infer_text_vl.append(embed.detach().cpu().numpy())

                ret_dict = multimodal_encoder.infer_text_ft(batch)
                for embed in ret_dict["cls_feats"].unbind():
                    text_embeddings_infer_text_ft.append(embed.detach().cpu().numpy())
                # ret_dict = model.infer_image(batch, mask_image=False)
                # for embed in ret_dict['image_feats'].unbind():
                #     image_embeddings_infer_image.append(embed[0].detach().cpu().numpy())  # shape: 197,768
                #
                # for embed in ret_dict['cls_vlffn_feats'].unbind():
                #     image_embeddings_infer_image_vl.append(embed.detach().cpu().numpy())  # shape: 197,768
                ret_dict = multimodal_encoder.infer_image_ft(batch)
                for embed in ret_dict["cls_feats"].unbind():
                    image_embeddings_infer_image_ft.append(embed.detach().cpu().numpy())

                # ret_dict = model.infer(batch, mask_text=False, mask_image=False)
                # for embed in ret_dict['text_feats'].unbind():
                #     text_embeddings.append(embed[0].detach().cpu().numpy())  # shape: 40,768
                # # text_embeddings.append(ret_dict['text_feats'].detach().cpu().numpy().concatenate())
                # for embed in ret_dict['image_feats'].unbind():
                #     image_embeddings.append(embed[0].detach().cpu().numpy())  # shape: 197,768
                #
                # for embed in ret_dict['cls_feats'].unbind():
                #     cls_features.append(embed.detach().cpu().numpy())  # shape: (768,)
                # for embed in ret_dict['raw_cls_feats'].unbind():
                #     raw_cls_features.append(embed.detach().cpu().numpy()) # shape: (768,)
                # cls_features.append(ret_dict['cls_feats'].detach().cpu().numpy())
                # raw_cls_features.append(ret_dict['raw_cls_feats'].detach().cpu().numpy())

            return torch.FloatTensor(image_embeddings_infer_image_ft).to(
                    self.device), torch.FloatTensor(text_embeddings_infer_text_ft).to(
                    self.device)
        else:
            print(f'Not implemented yet for this model'
                  f'{config["multi_modal_encoder"]}')
            raise NotImplementedError

    def update_embeddings(self):

        self.v_feat, self.t_feat = self.build_embeds(self.config,
                                                     self.multi_modal_encoder,
                                                     self.multimodal_data_loader)