import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from models.vlmo.dist_utils import all_gather


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]



    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    if pl_module.distill:
        mlm_logits_m = pl_module.mlm_score_m(infer["text_feats_m"])
        mlm_loss_m = F.cross_entropy(
            mlm_logits_m.view(-1, pl_module.config["vocab_size"]),
            mlm_labels.view(-1),
            ignore_index=-100,
        )
        mlm_loss = (1 - pl_module.alpha) * mlm_loss + pl_module.alpha * mlm_loss_m

    ret = {
        "mlm_loss": mlm_loss * 0.25,
        # "mlm_logits": mlm_logits,
        # "mlm_labels": mlm_labels,
        # "mlm_ids": infer["text_ids"],
    }

    if pl_module.config['compute_metrics']:
        phase = "train" if pl_module.training else "val"
        loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
        acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
            ret["mlm_logits"], ret["mlm_labels"]
        )
        ret['mlm_acc'] = acc
    # pl_module.log(f"mlm/{phase}/loss", loss)
    # pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_itm_hardneg(pl_module, batch, sim_i2t, sim_t2i):
    pos_len = batch["text_ids"].size(0)
    neg_len = batch["text_ids"].size(0)
    bsz = batch["text_ids"].size(0)
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len), torch.zeros(neg_len)]).to(
        pl_module.config['device']
    )

    batch = {k: v for k, v in batch.items()}
    infer_pos = pl_module.infer(batch, mask_text=False, mask_image=False)

    batch_text_ids = infer_pos["text_ids"]
    batch_text_masks = infer_pos["text_masks"]
    batch_image = infer_pos["image"]

    with torch.no_grad():
        # We gather tensors from all gpus to get more hard negative candidates.
        # cluster check
        if pl_module.config['cluster']:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # We gather tensors from all gpus to get more hard negative candidates.
        gathered_text_ids = [
            torch.zeros_like(batch_text_ids) for _ in range(world_size)
        ]
        gathered_text_masks = [
            torch.zeros_like(batch_text_masks) for _ in range(world_size)
        ]
        gathered_image = [
            torch.zeros_like(batch_image) for _ in range(world_size)
        ]

        if pl_module.config['cluster']:
            dist.all_gather(gathered_text_ids, batch_text_ids)
            dist.all_gather(gathered_text_masks, batch_text_masks)
            dist.all_gather(gathered_image, batch_image)

        all_text_ids = torch.cat(
            [batch_text_ids]
            + gathered_text_ids[:rank]
            + gathered_text_ids[rank + 1:]
        )
        all_text_masks = torch.cat(
            [batch_text_masks]
            + gathered_text_masks[:rank]
            + gathered_text_masks[rank + 1:]
        )
        all_image = torch.cat(
            [batch_image]
            + gathered_image[:rank]
            + gathered_image[rank + 1:]
        )

    with torch.no_grad():
        weights_i2t = F.softmax(sim_i2t[:bsz, :].float(), dim=1)
        weights_t2i = F.softmax(sim_t2i[:bsz, :].float(), dim=1)

        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)

    images_neg = []
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        images_neg.append(all_image[neg_idx])
    images_neg = torch.stack(images_neg, dim=0)

    # select a negative text for each image
    text_ids_neg = []
    text_masks_neg = []
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_ids_neg.append(all_text_ids[neg_idx])
        text_masks_neg.append(all_text_masks[neg_idx])

    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_masks_neg = torch.stack(text_masks_neg, dim=0)

    # text_labels is not used in ITM loss
    batch_imgs_neg = {"image": [images_neg], "text_ids": batch["text_ids"], "text_labels": batch["text_labels"],
                      "text_masks": batch["text_masks"]}
    infer_imags_neg = pl_module.infer(batch_imgs_neg, mask_text=False, mask_image=False)

    batch_text_neg = {"image": batch["image"], "text_ids": text_ids_neg, "text_labels": batch["text_labels"],
                      "text_masks": text_masks_neg}
    infer_text_neg = pl_module.infer(batch_text_neg, mask_text=False, mask_image=False)

    all_cls_feats = torch.cat([infer_pos["cls_feats"], infer_imags_neg["cls_feats"], infer_text_neg["cls_feats"]],
                              dim=0)

    itm_logits = pl_module.itm_score(all_cls_feats)
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    # if pl_module.distill:
    #     with torch.no_grad():



    ret = {
        "itm_loss": itm_loss,
        # "itm_logits": itm_logits,
        # "itm_labels": itm_labels,
    }

    if pl_module.config['compute_metrics']:
        phase = "train" if pl_module.training else "val"
        loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
        acc = getattr(pl_module, f"{phase}_itm_accuracy")(
            ret["itm_logits"], ret["itm_labels"]
        )
        ret['itm_acc'] = acc
    # pl_module.log(f"itm/{phase}/loss", loss)
    # pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


# The implementation of image-text contrastive refers to open_clip (https://github.com/mlfoundations/open_clip)
def compute_itc(pl_module, batch, aggregate=False):
    # pl_module.logit_scale.data = torch.clamp(pl_module.logit_scale.data, 0, 4.6052)

    infer_imag = pl_module.infer_image(batch, mask_image=False)
    infer_text = pl_module.infer_text(batch, mask_text=False)

    image_features = infer_imag["cls_feats"]
    text_features = infer_text["cls_feats"]
    logit_scale = pl_module.logit_scale.exp().mean()

    image_vlffn_features = infer_imag["cls_vlffn_feats"]
    text_vlffn_features = infer_text["cls_vlffn_feats"]
    logit_vl_scale = pl_module.logit_vl_scale.exp().mean()



    if aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1:]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1:]
        )


        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

        gathered_image_vlffn_features = [
            torch.zeros_like(image_vlffn_features) for _ in range(world_size)
        ]
        gathered_text_vlffn_features = [
            torch.zeros_like(text_vlffn_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_vlffn_features, image_vlffn_features)
        dist.all_gather(gathered_text_vlffn_features, text_vlffn_features)

        all_image_vlffn_features = torch.cat(
            [image_vlffn_features]
            + gathered_image_vlffn_features[:rank]
            + gathered_image_vlffn_features[rank + 1:]
        )
        all_text_vlffn_features = torch.cat(
            [text_vlffn_features]
            + gathered_text_vlffn_features[:rank]
            + gathered_text_vlffn_features[rank + 1:]
        )
        # this is needed to send gradients back everywhere.
        logits_per_vlffn_image = logit_vl_scale * all_image_vlffn_features @ all_text_vlffn_features.t()
        logits_per_vlffn_text = logits_per_vlffn_image.t()

    else:
        all_image_vlffn_features = image_vlffn_features
        all_text_vlffn_features = text_vlffn_features
        logits_per_vlffn_image = logit_vl_scale * all_image_vlffn_features @ all_text_vlffn_features.t()
        logits_per_vlffn_text = logits_per_vlffn_image.t()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long().to(device=logits_per_image.get_device())

    itc_loss = (
                       F.cross_entropy(logits_per_image.float(), ground_truth)
                       + F.cross_entropy(logits_per_text.float(), ground_truth)
               ) / 2

    itc_vlffn_loss = (
                             F.cross_entropy(logits_per_vlffn_image.float(), ground_truth)
                             + F.cross_entropy(logits_per_vlffn_text.float(), ground_truth)
                     ) / 2

    itc_total_loss = (itc_loss + itc_vlffn_loss) * 0.5

    if pl_module.distill:
        with torch.no_grad():
            image_features_m = infer_imag["cls_feats_m"]
            text_features_m = infer_text["cls_feats_m"]
            image_vlffn_features_m = infer_imag["cls_vlffn_feats_m"]
            text_vlffn_features_m = infer_text["cls_vlffn_feats_m"]
            image_features_m_all = torch.cat([image_features_m.t(), pl_module.image_queue.clone().detach()], dim=1)
            text_features_m_all = torch.cat([text_features_m.t(), pl_module.text_queue.clone().detach()], dim=1)
            image_vlffn_features_m_all = torch.cat([image_vlffn_features_m.t(), pl_module.image_vlffn_queue.clone().detach()], dim=1)
            text_vlffn_features_m_all = torch.cat([text_vlffn_features_m.t(), pl_module.text_vlffn_queue.clone().detach()], dim=1)

            logits_per_image_m = logit_scale * image_features_m @ text_features_m_all / pl_module.temp
            logits_per_text_m = logit_scale * text_features_m @ image_features_m_all / pl_module.temp

            logits_per_vlffn_image_m = logit_vl_scale * image_vlffn_features_m @ text_vlffn_features_m_all / pl_module.temp
            logits_per_vlffn_text_m = logit_vl_scale * text_vlffn_features_m @ image_vlffn_features_m_all / pl_module.temp

            ground_truth_m = torch.arange(len(logits_per_image_m)).long().to(device=logits_per_image_m.get_device())

            itc_loss_m = (
                                 F.cross_entropy(logits_per_image_m.float(), ground_truth_m)
                                    + F.cross_entropy(logits_per_text_m.float(), ground_truth_m)
                            ) / 2

            itc_vlffn_loss_m = (
                                        F.cross_entropy(logits_per_vlffn_image_m.float(), ground_truth_m)
                                        + F.cross_entropy(logits_per_vlffn_text_m.float(), ground_truth_m)
                                ) / 2

            itc_total_loss_m = (itc_loss_m + itc_vlffn_loss_m) * 0.5

            itc_total_loss = pl_module.alpha * itc_total_loss_m + itc_total_loss * (1 - pl_module.alpha)

            pl_module._dequeue_and_enqueue(image_features_m, text_features_m,
                                           image_vlffn_features_m, text_vlffn_features_m)

    ret = {
        "itc_loss": itc_total_loss,
        "itc_i2t_logits": logits_per_image,
        "itc_t2i_logits": logits_per_text,
        # "itc_labels": ground_truth,
        # "itc_logit_scale": logit_scale,
        # "itc_logit_vl_scale": logit_vl_scale,
    }

    if pl_module.config['compute_metrics']:
        phase = "train" if pl_module.training else "val"
        loss = getattr(pl_module, f"{phase}_itc_loss")(ret["itc_loss"])
        scale = getattr(pl_module, f"{phase}_itc_logit_scale")(ret["itc_logit_scale"])
        i2t_acc = getattr(pl_module, f"{phase}_itc_i2t_accuracy")(
            ret["itc_i2t_logits"], ret["itc_labels"]
        )
        t2i_acc = getattr(pl_module, f"{phase}_itc_t2i_accuracy")(
            ret["itc_t2i_logits"], ret["itc_labels"]
        )
        ret['i2t_acc'] = i2t_acc
        ret['t2i_acc'] = t2i_acc

        vl_scale = getattr(pl_module, f"{phase}_itc_vl_logit_scale")(ret["itc_logit_vl_scale"])

        vl_i2t_acc = getattr(pl_module, f"{phase}_itc_vl_i2t_accuracy")(
            logits_per_vlffn_image, ret["itc_labels"]
        )
        vl_t2i_acc = getattr(pl_module, f"{phase}_itc_vl_t2i_accuracy")(
            logits_per_vlffn_text, ret["itc_labels"]
        )
        ret['vl_i2t_acc'] = vl_i2t_acc
        ret['vl_t2i_acc'] = vl_t2i_acc


    return ret



def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


