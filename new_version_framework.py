import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ultralytics import YOLO

# =============================== Encoders =================================
class ImageEncoder(nn.Module):
    """
    Unified image encoder abstraction:
    - ResNet backbone (multi-scale conv features)
    - ViT backbone (patch embeddings projected)
    - YOLOv8 backbone (multi-scale conv features)
    Always returns: [HV1, HV2, HV3] with shapes:
      HV1: (B, 49, D1)
      HV2: (B, 196, D2)
      HV3: (B, 784, D3)
    """
    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        self.backbone = backbone.lower()
        self.out_dims = [2048, 1536, 1024]

        if self.backbone.startswith("resnet"):
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
            modules = list(resnet.children())[:-2]
            self.encoder = nn.Sequential(*modules)
            self.proj2 = nn.Linear(2048, 1536)
            self.proj3 = nn.Linear(2048, 1024)

        elif self.backbone == "vit_b_16":
            vit = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
            vit.heads = nn.Identity()
            self.encoder = vit
            self.proj1 = nn.Linear(768, 2048)
            self.proj2 = nn.Linear(768, 1536)
            self.proj3 = nn.Linear(768, 1024)

        elif self.backbone.startswith("yolov8"):
            self.encoder = YOLO(f"{self.backbone}.pt").model.backbone
            self.proj1, self.proj2, self.proj3 = None, None, None  # will be built dynamically

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, images):
        if self.backbone.startswith("resnet"):
            fmap = self.encoder(images)  # (B, 2048, H/32, W/32)

            hv1 = F.adaptive_avg_pool2d(fmap, (7, 7)).flatten(2).transpose(1, 2)
            hv2 = self.proj2(
                F.adaptive_avg_pool2d(fmap, (14, 14)).flatten(2).transpose(1, 2)
            )
            hv3 = self.proj3(
                F.adaptive_avg_pool2d(fmap, (28, 28)).flatten(2).transpose(1, 2)
            )
            return [hv1, hv2, hv3]

        elif self.backbone == "vit_b_16":
            patch_tokens = self.encoder._process_input(images)  # (B, N, 768)
            hv1 = self.proj1(patch_tokens[:, :49, :])
            hv2 = self.proj2(patch_tokens[:, :196, :])
            hv3 = self.proj3(patch_tokens[:, :784, :])
            return [hv1, hv2, hv3]

        elif self.backbone.startswith("yolov8"):
            feats = self.encoder(images)  # list of feature maps
            # build proj layers dynamically if first call
            if self.proj1 is None:
                c1, c2, c3 = feats[2].shape[1], feats[1].shape[1], feats[0].shape[1]
                self.proj1 = nn.Linear(c1, 2048).to(feats[2].device)
                self.proj2 = nn.Linear(c2, 1536).to(feats[1].device)
                self.proj3 = nn.Linear(c3, 1024).to(feats[0].device)

            hv1 = self.proj1(feats[2].flatten(2).transpose(1, 2))
            hv2 = self.proj2(feats[1].flatten(2).transpose(1, 2))
            hv3 = self.proj3(feats[0].flatten(2).transpose(1, 2))
            return [hv1, hv2, hv3]


class TextEncoder(nn.Module):
    """
    Wraps a HuggingFace seq2seq model (e.g., BART).
    Returns contextual embeddings (HT), input embeddings (E), and mask.
    """
    def __init__(self, model_name="facebook/bart-base", max_length=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.hidden_size = self.model.config.d_model
        self.max_length = max_length

    def forward(self, texts, aspects):
        # prepend aspect for aspect-aware input
        joint = [f"{a} {self.tokenizer.sep_token} {t}" for a, t in zip(aspects, texts)]
        enc = self.tokenizer(
            joint, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids, mask = enc["input_ids"].to(self.model.device), enc["attention_mask"].to(self.model.device)

        outputs = self.model.model.encoder(input_ids=input_ids, attention_mask=mask, return_dict=True)
        HT = outputs.last_hidden_state  # contextual features (B, T, d_model)

        E = self.model.model.shared(input_ids)  # embeddings (B, T, d_model)

        return HT, E, mask

# =============================== Attention Blocks ============================
class SimpleSelfAttentionBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_proj = self.proj(x)
        attn_out, _ = self.attn(x_proj, x_proj, x_proj)
        return self.norm(x_proj + self.drop(attn_out))


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, v_dim, t_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, hidden_dim)
        self.t_proj = nn.Linear(t_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, HV, HT, text_mask=None):
        HVp, HTp = self.v_proj(HV), self.t_proj(HT)
        key_padding_mask = (text_mask == 0) if text_mask is not None else None
        attn_out, _ = self.attn(HVp, HTp, HTp, key_padding_mask=key_padding_mask)
        return self.norm(HVp + self.drop(attn_out))

# =============================== CRD =========================================
class ConditionalRelationDetector(nn.Module):
    def __init__(self, visual_dims=(2048,1536,1024), text_dim=768, hidden_dim=512):
        super().__init__()
        self.num_scales = 3
        self.self_blocks = nn.ModuleList([SimpleSelfAttentionBlock(d, hidden_dim) for d in visual_dims])
        self.cross_blocks = nn.ModuleList([CrossModalAttentionBlock(hidden_dim, text_dim, hidden_dim) for _ in range(3)])
        self.class_heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(3)])
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, visuals, text_features, text_mask=None, labels_crd=None):
        logits_list, probs_list, filtered = [], [], []

        for i in range(self.num_scales):
            HVi = visuals[i]
            HVi1 = self.self_blocks[i](HVi)
            HVi2 = self.cross_blocks[i](HVi1, text_features, text_mask)
            Hmax, _ = torch.max(HVi2, dim=1)
            logits = self.class_heads[i](Hmax)
            probs = F.softmax(logits, dim=-1)

            Gi = probs[:,1].unsqueeze(-1).unsqueeze(-1).expand_as(HVi2)
            HVi3 = Gi * HVi2

            logits_list.append(logits)
            probs_list.append(probs)
            filtered.append(HVi3)

        crd_logits = torch.stack(logits_list, dim=1)  # (B,3,2)
        crd_probs  = torch.stack(probs_list, dim=1)

        loss_crd = None
        if labels_crd is not None:
            loss_crd = self.loss_fn(crd_logits.view(-1,2), labels_crd.view(-1))

        return crd_logits, crd_probs, filtered, loss_crd

# =============================== VOL =========================================
class DetectHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_classes=3, num_anchors=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_anchors * (4 + 1 + num_classes))
        )
        self.num_anchors, self.num_classes = num_anchors, num_classes

    def forward(self, x):
        B, S, D = x.shape
        out = self.fc(x).view(B, S, self.num_anchors, 4+1+self.num_classes)
        return {
            "boxes": out[..., :4],
            "obj_logits": out[..., 4],
            "class_logits": out[..., 5:]
        }


class VOLLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, preds, targets):
        loc_loss = nn.MSELoss(reduction="none")
        cls_loss = nn.CrossEntropyLoss(reduction="none")
        obj_loss = nn.BCEWithLogitsLoss(reduction="none")

        pb, po, pc = preds["boxes"], preds["obj_logits"], preds["class_logits"]
        gb, go, gc = targets["boxes"], targets["obj_mask"], targets["classes"]

        box_loss = loc_loss(pb, gb).mean(-1)
        LLOC = (box_loss * go).sum() / (go.sum() + 1e-6)

        if (go > 0.5).any():
            Lcls = cls_loss(pc[go>0.5], gc[go>0.5]).mean()
        else:
            Lcls = torch.tensor(0.0, device=pc.device)

        Lobj = obj_loss(po, go.float()).mean()
        return {"LLOC": LLOC, "LCLS": Lcls+Lobj}


class VisualOpinionLearner(nn.Module):
    def __init__(self, hidden_dim=512, text_dim=768, backbone="resnet50"):
        super().__init__()
        self.backbone = backbone
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()
        if backbone.startswith("resnet") or backbone.startswith("vit"):
            self.det_heads = nn.ModuleList([DetectHead(hidden_dim, hidden_dim) for _ in range(3)])
        else:
            self.det_heads = None

        self.W_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.WA = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Linear(hidden_dim*2, 1) for _ in range(3)])
        self.attn_blocks = nn.ModuleList([CrossModalAttentionBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)])
        self.loss_fn = VOLLoss()

    def forward(self, visuals, text_features, text_mask=None, targets=None):
        HT = self.text_proj(text_features)
        H_hats, total_loss = [], None

        if targets is not None and self.det_heads is not None:
            total_loss = {"LLOC":0.0, "LCLS":0.0}
            for i,HVi in enumerate(visuals):
                preds = self.det_heads[i](self.W_proj[i](HVi))
                det_loss = self.loss_fn(preds, targets[i])
                total_loss["LLOC"] += det_loss["LLOC"]
                total_loss["LCLS"] += det_loss["LCLS"]

        for i,HVi in enumerate(visuals):
            Hv_proj = self.W_proj[i](HVi)
            Hv_att = self.attn_blocks[i](HVi, HT, text_mask)
            HvA_proj = self.WA[i](Hv_att)
            alpha = torch.sigmoid(self.gates[i](torch.cat([Hv_proj, HvA_proj], dim=-1)))
            H_hats.append(alpha*Hv_proj + (1-alpha)*HvA_proj)

        return H_hats, total_loss

# =============================== MSA =========================================
class MultiModalSentimentAnalyzer(nn.Module):
    def __init__(self, llm_model, hidden_dim=512):
        super().__init__()
        self.llm = llm_model
        self.d_model = llm_model.config.d_model
        self.vocab_size = llm_model.config.vocab_size
        self.WV2 = nn.Linear(196, 49)
        self.WV3 = nn.Linear(784, 49)
        self.fuse_proj = nn.Linear(hidden_dim*3, self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, H_hats, E, text_mask, decoder_input_ids, labels=None):
        HV1, HV2, HV3 = H_hats
        HV2_49 = self.WV2(HV2.transpose(1,2)).transpose(1,2)
        HV3_49 = self.WV3(HV3.transpose(1,2)).transpose(1,2)
        H_cat = torch.cat([HV1, HV2_49, HV3_49], dim=-1)
        H_vis = self.fuse_proj(H_cat)

        enc_in = torch.cat([H_vis, E], dim=1)
        vis_mask = torch.ones(H_vis.size(0), H_vis.size(1), device=E.device, dtype=torch.long)
        enc_mask = torch.cat([vis_mask, text_mask], dim=1)

        enc_out = self.llm.model.encoder(inputs_embeds=enc_in,
                                         attention_mask=enc_mask,
                                         return_dict=True)
        dec_out = self.llm.model.decoder(input_ids=decoder_input_ids,
                                         encoder_hidden_states=enc_out.last_hidden_state,
                                         encoder_attention_mask=enc_mask,
                                         return_dict=True)
        logits = self.lm_head(dec_out.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss


class CORSA(nn.Module):
    """
    Full pipeline: ImageEncoder → CRD → VOL → MSA.
    Implements joint loss: λ_D * L_CRD + λ_L*(L_LOC+L_CLS) + L_MSA
    """
    def __init__(self, backbone="resnet50", text_model="facebook/bart-base",
                 hidden_dim=512, lambda_D=1.0, lambda_L=1.0):
        super().__init__()
        # components
        self.image_encoder = ImageEncoder(backbone=backbone)
        self.text_encoder = TextEncoder(model_name=text_model)
        self.crd = ConditionalRelationDetector(
            visual_dims=(2048,1536,1024),
            text_dim=self.text_encoder.hidden_size,
            hidden_dim=hidden_dim
        )
        self.vol = VisualOpinionLearner(hidden_dim=hidden_dim,
                                        text_dim=self.text_encoder.hidden_size,
                                        backbone=backbone)
        self.msa = MultiModalSentimentAnalyzer(self.text_encoder.model,
                                               hidden_dim=hidden_dim)
        # loss weights
        self.lambda_D = lambda_D
        self.lambda_L = lambda_L

    def forward(self, images, texts, aspects,
                decoder_input_ids, labels_msa,
                labels_crd=None, targets_vol=None):
        """
        Args:
            images: tensor (B,3,H,W)
            texts: list of B strings
            aspects: list of B strings
            decoder_input_ids: (B, T_dec)
            labels_msa: (B, T_dec) sentiment targets
            labels_crd: (B,3) ground truth relevance labels (optional)
            targets_vol: list of length B with detection targets (optional)
        Returns:
            total_loss, dict of sub-losses, msa_logits
        """
        # 1. Encode
        visuals = self.image_encoder(images)        # [HV1, HV2, HV3]
        HT, E, mask = self.text_encoder(texts, aspects)

        # 2. CRD
        crd_logits, crd_probs, filtered_visuals, loss_crd = self.crd(
            visuals, HT, text_mask=mask, labels_crd=labels_crd
        )

        # 3. VOL
        H_hats, loss_vol = self.vol(filtered_visuals, HT, text_mask=mask, targets=targets_vol)

        # 4. MSA
        msa_logits, loss_msa = self.msa(H_hats, E, mask,
                                        decoder_input_ids, labels=labels_msa)

        # 5. Aggregate joint loss
        total_loss = 0.0
        if loss_crd is not None:
            total_loss += self.lambda_D * loss_crd
        if loss_vol is not None:
            total_loss += self.lambda_L * (loss_vol["LLOC"] + loss_vol["LCLS"])
        if loss_msa is not None:
            total_loss += loss_msa

        return total_loss, {
            "L_CRD": loss_crd,
            "L_VOL": loss_vol,
            "L_MSA": loss_msa,
            "total": total_loss
        }, msa_logits


def demo_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CORSA(backbone="resnet50", text_model="facebook/bart-base").to(device)

    B = 2
    images = torch.randn(B, 3, 224, 224).to(device)
    texts = ["this laptop has a good screen", "the camera quality is bad"]
    aspects = ["screen", "camera"]

    # fake decoder inputs (teacher forcing) and labels
    T_dec = 5
    decoder_input_ids = torch.randint(0, model.text_encoder.model.config.vocab_size,
                                      (B, T_dec), device=device)
    labels_msa = torch.randint(0, model.text_encoder.model.config.vocab_size,
                               (B, T_dec), device=device)

    # fake CRD labels (all scales relevant)
    labels_crd = torch.ones(B, 3, dtype=torch.long, device=device)

    # fake VOL targets (empty dicts for now)
    targets_vol = [{"boxes": torch.zeros(B,49,1,4,device=device),
                    "obj_mask": torch.zeros(B,49,1,device=device),
                    "classes": torch.zeros(B,49,1,dtype=torch.long,device=device)}
                   for _ in range(3)]

    total_loss, losses, logits = model(
        images, texts, aspects,
        decoder_input_ids, labels_msa,
        labels_crd=labels_crd, targets_vol=targets_vol
    )

    print("Total loss:", total_loss)
    print("Loss dict:", {k: (v if v is None else float(v)) for k,v in losses.items()})
    print("MSA logits shape:", logits.shape)


if __name__ == "__main__":
    demo_forward()