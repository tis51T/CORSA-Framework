import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoTokenizer, AutoModel
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# =============================== Encoders =================================
class ImageEncoder(nn.Module):
    """
    Unified image encoder abstraction:
    - YOLOv8 backbone (multi-scale conv features)
    - ResNet backbone (pooled + projected multi-scale features)
    - ViT backbone (patch embeddings projected to YOLO-like dims)
    Always returns: [HV1, HV2, HV3] as (B, S, D)
    """
    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        self.backbone = backbone.lower()
        self.out_dims = [2048, 1536, 1024]

        if self.backbone.startswith("resnet"):
            # safer resnet weights selection
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if self.backbone == "resnet50" else None
                resnet = models.resnet50(weights=weights)
            else:
                resnet = models.resnet50(weights=None)
            modules = list(resnet.children())[:-2]
            self.encoder = nn.Sequential(*modules)

            # project to YOLO-like dims
            self.proj2 = nn.Linear(2048, 1536)
            self.proj3 = nn.Linear(2048, 1024)

        elif self.backbone == "vit_b_16":
            vit = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
            vit.heads = nn.Identity()
            self.encoder = vit

            # project 768 → YOLO-like dims
            self.proj1 = nn.Linear(768, 2048)
            self.proj2 = nn.Linear(768, 1536)
            self.proj3 = nn.Linear(768, 1024)

        elif self.backbone.startswith("yolov8"):
            self.encoder = YOLO(f"{self.backbone}.pt").model.backbone
            self.proj1 = nn.Linear(256, 2048)   # deepest scale → 2048
            self.proj2 = nn.Linear(128, 1536)   # mid scale   → 1536
            self.proj3 = nn.Linear(64, 1024)    # shallow     → 1024

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, images):
        if self.backbone.startswith("resnet"):
            feature_map = self.encoder(images)  # (B, 2048, W/32, H/32)

            hv1 = F.adaptive_avg_pool2d(feature_map, (7, 7)).flatten(2).transpose(1, 2)
            hv2 = F.adaptive_avg_pool2d(feature_map, (14, 14)).flatten(2).transpose(1, 2)
            hv2 = self.proj2(hv2)
            hv3 = F.adaptive_avg_pool2d(feature_map, (28, 28)).flatten(2).transpose(1, 2)
            hv3 = self.proj3(hv3)

            return [hv1, hv2, hv3]

        elif self.backbone == "vit_b_16":
            patch_tokens = self.encoder._process_input(images)  # (B, N, 768)
            B, S, D = patch_tokens.shape

            hv1 = patch_tokens[:, :49, :]
            hv2 = patch_tokens[:, :196, :]
            hv3 = patch_tokens[:, :784, :]

            hv1 = self.proj1(hv1)
            hv2 = self.proj2(hv2)
            hv3 = self.proj3(hv3)

            return [hv1, hv2, hv3]

        elif self.backbone.startswith("yolov8"):
            features = self.encoder(images)  # list of feature maps
            f1 = features[2]   # deeper feature map
            hv1 = f1.flatten(2).transpose(1, 2)  # (B, S, C)
            hv1 = self.proj1(hv1)

            # HV2: mid map (14x14 → 196 tokens), project → 1536
            f2 = features[1]
            hv2 = f2.flatten(2).transpose(1, 2)
            hv2 = self.proj2(hv2)

            # HV3: largest map (28x28 → 784 tokens), project → 1024
            f3 = features[0]
            hv3 = f3.flatten(2).transpose(1, 2)
            hv3 = self.proj3(hv3)

            return [hv1, hv2, hv3]

class TextEncoder(nn.Module):
    """
    Wrapper for HuggingFace transformer encoder (BERT, BART, RoBERTa, etc.)
    Outputs contextual embeddings for tokens.
    """
    def __init__(self, model_name="bert-base-uncased", max_length=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.max_length = max_length

    def forward(self, texts, aspects):
        """
        Args:
            texts: list of strings
            aspects: list of aspect terms (strings)
        Returns:
            HT: tensor (B, T, D_t)
            mask: attention mask (B, T)
        """
        # Aspect-aware encoding: prepend aspect
        joint_texts = [f"{a} {self.tokenizer.sep_token} {t}" for a,t in zip(aspects, texts)]
        enc = self.tokenizer(joint_texts, ..., return_tensors="pt")
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        HT = outputs.last_hidden_state  # (B, T, D_t)
        Wemb = self.model.get_input_embeddings()(input_ids)  # (B, T, D_emb)
        return HT, Wemb, attention_mask

# =============================== Attention Blocks ============================
class SimpleSelfAttentionBlock(nn.Module):
    """Lightweight self-attention for visual features"""
    def __init__(self, in_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim) # WVi
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, S, D_in)
        x_proj = self.proj(x)  # (B, S, hidden_dim)
        attn_out, _ = self.attn(x_proj, x_proj, x_proj)
        # may be have FFN here
        return self.norm(x_proj + self.drop(attn_out))

class CrossModalAttentionBlock(nn.Module):
    """Cross-modal attention: visual queries attend to text keys/values"""
    def __init__(self, v_dim, t_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, hidden_dim)
        self.t_proj = nn.Linear(t_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, HV, HT, text_mask=None):
        """
        HV: (B, Sv, Dv) visual features
        HT: (B, St, Dt) text features
        text_mask: (B, St) with 1=valid, 0=pad
        """
        HV_proj = self.v_proj(HV)
        HT_proj = self.t_proj(HT)

        # convert HuggingFace attention_mask → key_padding_mask
        # key_padding_mask: True = ignore, False = keep
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = (text_mask == 0)  # invert

        attn_out, _ = self.attn(query=HV_proj,
                                key=HT_proj,
                                value=HT_proj,
                                key_padding_mask=key_padding_mask)
        return self.norm(HV_proj + self.drop(attn_out))

# ================================= CRD block =================================
class ConditionalRelationDetector(nn.Module):
    def __init__(self, visual_dims=(2048,1536,1024), text_dim=768, hidden_dim=512, num_heads=8, dropout=0.1):
        """
        Args:
            visual_dims: list of 3 ints [Dv1, Dv2, Dv3]
            text_dim: int (Dt, dimension of text features)
            hidden_dim: internal hidden dim
        """
        super().__init__()
        assert len(visual_dims) == 3, "Expect three visual scales."

        self.num_scales = 3

        # (1) self-attention blocks (visual only)
        self.self_blocks = nn.ModuleList([
            SimpleSelfAttentionBlock(in_dim=d, hidden_dim=hidden_dim, num_heads=num_heads,  dropout=dropout)
            for d in visual_dims
        ])

        # (2) cross-modal attention blocks (visual ↔ text)
        self.cross_blocks = nn.ModuleList([
            CrossModalAttentionBlock(v_dim=hidden_dim,
                                     t_dim=text_dim,
                                     hidden_dim=hidden_dim,
                                     num_heads=num_heads,
                                     dropout=dropout)
            for _ in range(3)
        ])

        # (4) classification heads (relevant / irrelevant)
        self.class_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(3)
        ])

        # # (5) project back
        # self.post_proj = nn.ModuleList([
        #     nn.Linear(hidden_dim, d) for d in visual_dims
        # ])

    def forward(self, visuals, text_features, text_mask=None):
        """
        Args:
            visuals: list [HV1, HV2, HV3] with shapes [(B, S1, Dv1), ...]
            text_features: (B, T, Dt)
            text_mask: (B, T) where 1=valid, 0=pad

        Returns:
            crd_logits: (B, 3, 2) logits for relevance
            crd_probs: (B, 3, 2) softmax probs
            filtered_visuals: list of filtered visuals [H'''_V1, H'''_V2, H'''_V3]
        """
        B = text_features.size(0)
        logits_list, probs_list, HV_triple_prime = [], [], []

        for i in range(self.num_scales):
            HVi = visuals[i]  # (B, Si, Dvi)
            # (1) self-attention → hidden_dim
            HVi_prime = self.self_blocks[i](HVi)  
            # (2) cross-modal attention with text
            HVi_double_prime = self.cross_blocks[i](HVi_prime, text_features, text_mask=text_mask)
            # (3) max-pool across sequence (Si)
            HVi_max, _ = torch.max(HVi_double_prime, dim=1)  # (B, hidden_dim)
            # (4) classify
            logits = self.class_heads[i](HVi_max)          # (B, 2)
            probs = F.softmax(logits, dim=-1)           # (B, 2)
            # (5) filter matrix Gi using relevant prob
            prob_relevant = probs[:, 1].unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
            Gi = prob_relevant.expand(-1, HVi_double_prime.size(1), HVi_double_prime.size(2))  # (B,Si,H)
            HVi_triple_prime  = Gi * HVi_double_prime  # (B, Si, H) filtered features
            logits_list.append(logits)
            probs_list.append(probs)

            # # (6) optional: project back to original dim
            # HVi_triple_prime = self.post_proj[i](HVi_triple_prime)
            HV_triple_prime.append(HVi_triple_prime)

        self.W_Vi = [self_block.proj for self_block in self.self_blocks]
        crd_logits = torch.stack(logits_list, dim=1)  # (B, 3, 2)
        crd_probs  = torch.stack(probs_list, dim=1)   # (B, 3, 2)

        # make sure no zero probabilities to avoid log(0)
        relevant_probs = crd_probs[:, :, 1]  # shape: (B, 3)

        # clamp to avoid log(0)
        eps = 1e-12
        relevant_probs = relevant_probs.clamp(min=eps)

        # compute log
        log_probs = torch.log(relevant_probs)  # (B, 3)

        # sum over scales i=1..3
        sum_log = log_probs.sum(dim=1)  # (B,)

        # average over batch and negate
        loss_crd = -sum_log.mean()

        return crd_logits, loss_crd, HV_triple_prime

# ================================= VOL block =================================
class DetectHead(nn.Module):
    """
    Simple detection head for VOL when using ResNet or ViT as image encoder.
    Input: (B, S, D)
    Output: dict with boxes, objectness, class_logits
    """
    def __init__(self, in_dim, hidden_dim=256, num_classes=3, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_anchors * (4 + 1 + num_classes))
        )

    def forward(self, x):
        """
        x: (B, S, D)
        returns:
          - boxes: (B, S, A, 4)
          - obj_logits: (B, S, A)
          - class_logits: (B, S, A, C)
        """
        B, S, D = x.shape
        out = self.fc(x)  # (B, S, A*(4+1+C))
        out = out.view(B, S, self.num_anchors, 4 + 1 + self.num_classes)

        boxes = out[..., :4]          # (B, S, A, 4)
        obj_logits = out[..., 4]      # (B, S, A)
        class_logits = out[..., 5:]   # (B, S, A, C)

        return {
            "boxes": boxes, "obj_logits": obj_logits, "class_logits": class_logits
        }

class VOLLoss(nn.Module):
    """
    Implements L_LOC and L_CLS from the paper.
    """
    def __init__(self):
        super().__init__()
        self.loc_loss = nn.MSELoss(reduction="none")
        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.obj_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targets):
        """
        preds: dict from SimpleDETHead
        targets: dict with keys:
            - boxes: (B, S, A, 4)
            - obj_mask: (B, S, A) 0/1
            - classes: (B, S, A) long
        """
        pred_boxes = preds["boxes"]
        pred_obj = preds["obj_logits"]
        pred_cls = preds["class_logits"]

        gt_boxes = targets["boxes"]
        gt_obj = targets["obj_mask"]
        gt_cls = targets["classes"]

        # L_LOC: only on positives
        box_loss = self.loc_loss(pred_boxes, gt_boxes).mean(-1)  # (B,S,A)
        LLOC = (box_loss * gt_obj).sum() / (gt_obj.sum() + 1e-6)

        # L_CLS: CE for classes + BCE for objectness
        pos_mask = gt_obj > 0.5
        if pos_mask.any():
            cls_loss = self.cls_loss(pred_cls[pos_mask], gt_cls[pos_mask])
            Lcls_term = cls_loss.mean()
        else:
            Lcls_term = torch.tensor(0.0, device=pred_cls.device)

        obj_loss = self.obj_loss(pred_obj, gt_obj.float()).mean()
        LCLS = Lcls_term + obj_loss

        return {"LLOC": LLOC, "LCLS": LCLS, "total": LLOC + LCLS}

class VisualOpinionLearner(nn.Module):
    def __init__(self, visual_dims=(2048,1536,1024), text_dim=768,
                 hidden_dim=512, num_heads=8, dropout=0.1, backbone="yolov8n"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.visual_dims = visual_dims
        self.text_dim = text_dim
        self.backbone = backbone.lower()

        # project text -> hidden once (registered module)
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()

        # If using feature-level DetectHead, its in_dim must match hidden_dim
        if backbone.startswith("yolov8"):
            # keep det_model None or a proper ultralytics model expecting images
            self.det_model = None
        elif backbone.startswith("resnet") or backbone.startswith("vit"):
            self.det_model = nn.ModuleList([DetectHead(in_dim=hidden_dim, hidden_dim=hidden_dim)
                                            for _ in range(3)])
        else:
            raise ValueError(f"Unsupported backbone for VOL: {backbone}")

        # projections
        self.W_triple_prime = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)]) # Dvi to H
        self.W_A = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)]) # H_A will have shape (B, Si, H)
        self.W_alpha = nn.ModuleList([nn.Linear(hidden_dim*2, 1) for _ in range(3)]) # scalar gate per token

        # cross-modal attention
        self.H_A = nn.ModuleList([
            CrossModalAttentionBlock(v_dim=hidden_dim, t_dim=hidden_dim, hidden_dim=hidden_dim,
                                     num_heads=num_heads, dropout=dropout)
            for _ in range(3)
        ])

    def forward(self, visuals, text_features, text_mask=None, targets=None):
        """
        visuals: list [H'''_V1,H'''_V2,H'''_V3], each (B,Si,Dvi)
        text_features: (B,T,Dt)
        targets: ground truth labels for YOLO (list of length B), optional
        """
        B = text_features.size(0)
        HT = self.text_proj(text_features)
        H_hats, total_loc_loss, total_cls_loss = [], 0.0, 0.0

        # --- detection step ---
        if targets is not None:
            if self.backbone in ["resnet", "vit"]:
                for i, HVi in enumerate(visuals):
                    preds = self.det_model[i](HVi)  # SimpleDETHead
                    assigned_targets = self.assign_targets(targets, S=HVi.size(1), A=preds["boxes"].size(2), num_classes=preds["class_logits"].size(-1))
                    det_loss = VOLLoss()(preds, assigned_targets)
                    total_loc_loss += det_loss["LLOC"]
                    total_cls_loss += det_loss["LCLS"]

            elif self.backbone.startswith("yolov8"):
                preds = self.det_model.detect(visuals, verbose=False)  # all scales together
                det_loss = self.det_model.loss(preds, targets)
                total_loc_loss += det_loss["box"]
                total_cls_loss += det_loss["cls"] + det_loss["dfl"]

        # --- cross-modal fusion ---
        for i, HVi in enumerate(visuals):
            HVi_project = self.W_triple_prime[i](HVi)
            HVi_A = self.H_A[i](HVi, HT, text_mask)
            HVi_A_project = self.W_A[i](HVi_A)

            gate_in = torch.cat([HVi_project, HVi_A_project], dim=-1)
            alpha = torch.sigmoid(self.W_alpha[i](gate_in))

            HVi_hat = alpha * HVi_project + (1 - alpha) * HVi_A_project
            H_hats.append(HVi_hat)

        loss = None
        if targets is not None:
            loss = {
                "LLOC": total_loc_loss / len(visuals),
                "LCLS": total_cls_loss / len(visuals),
                "total": (total_loc_loss + total_cls_loss) / len(visuals)
            }
        return H_hats, loss

    @staticmethod
    def assign_targets(gt, S, A, device):
        B = len(gt)
        grid_size = int(S ** 0.5)
        boxes = torch.zeros(B, S, A, 4, device=device)
        obj_mask = torch.zeros(B, S, A, device=device)
        classes = torch.zeros(B, S, A, dtype=torch.long, device=device)

        for b in range(B):
            for obj in gt[b]:
                cls, x, y, w, h = obj
                gx, gy = x * grid_size, y * grid_size
                gi = max(0, min(grid_size - 1, int(gx)))
                gj = max(0, min(grid_size - 1, int(gy)))
                cell_idx = gj * grid_size + gi
                boxes[b, cell_idx, 0] = torch.tensor([x, y, w, h], device=device)
                obj_mask[b, cell_idx, 0] = 1.0
                classes[b, cell_idx, 0] = int(cls)

        return {"boxes": boxes, "obj_mask": obj_mask, "classes": classes}

# =============================== Multi-modal Sentiment Analyzer =================================
class MultiModalSentimentAnalyzer(nn.Module):
    """
    MSA aligned to CORSA eq(11).
    - Expects H_hats = [H1 (B,49,H), H2 (B,196,H), H3 (B,784,H)]
    - E: text input embeddings (B, T_text, d_model) from the SAME BART embedding layer
    - text_mask: (B, T_text) attention mask (1 valid, 0 pad)
    - decoder_input_ids: (B, T_dec) shifted-right target ids (for teacher forcing)
    """
    def __init__(self, llm_model, hidden_dim=512, vocab_size=None):
        super().__init__()
        # llm_model: a HF seq2seq model (BartForConditionalGeneration or similar)
        self.llm = llm_model
        self.d_model = self.llm.config.d_model  # target dimension for encoder/decoder
        if vocab_size is None:
            vocab_size = self.llm.config.vocab_size

        # WV2: map 196 tokens -> 49 tokens, WV3: map 784 -> 49 (linear across token axis).
        self.WV2 = nn.Linear(196, 49)   # will be applied to HV2.transpose(1,2)
        self.WV3 = nn.Linear(784, 49)

        # fuse channel dims and project to d_model
        self.fuse_proj = nn.Linear(hidden_dim * 3, self.d_model)  # (concatenated channels -> d_model)

        # final token->vocab head
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=True)

        # loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # -100 default ignore

    def forward(self, H_hats, E, text_mask, decoder_input_ids, labels=None):
        # H_hats: list of tensors with shapes [ (B,49,H), (B,196,H), (B,784,H) ]
        HV1, HV2, HV3 = H_hats
        B = HV1.size(0)

        # map HV2 (B,196,H) -> (B,49,H)
        # apply WV2 along token axis: transpose -> (B,H,196) -> linear -> (B,H,49) -> transpose back
        HV2_49 = self.WV2(HV2.transpose(1,2)).transpose(1,2)  # (B,49,H)
        HV3_49 = self.WV3(HV3.transpose(1,2)).transpose(1,2)  # (B,49,H)

        # concat channels: (B,49, 3*H)
        H_cat = torch.cat([HV1, HV2_49, HV3_49], dim=-1)

        # project channels to d_model => (B,49,d_model)
        H_vis = self.fuse_proj(H_cat)

        # concatenate visual tokens with text embeddings along sequence dimension
        # E is expected to be (B, T_text, d_model)
        enc_inputs = torch.cat([H_vis, E], dim=1)  # (B, 49 + T_text, d_model)

        # build encoder attention mask: 1 for visual tokens (all valid), then original text_mask
        device = enc_inputs.device
        vis_mask = torch.ones(B, H_vis.size(1), dtype=torch.long, device=device)
        enc_attention_mask = torch.cat([vis_mask, text_mask], dim=1)  # (B, 49 + T_text)

        # call encoder (pass inputs_embeds)
        encoder_outputs = self.llm.model.encoder(inputs_embeds=enc_inputs,
                                                 attention_mask=enc_attention_mask,
                                                 return_dict=True)
        encoder_hidden = encoder_outputs.last_hidden_state  # (B, seq_len_enc, d_model)

        # call decoder with encoder_hidden_states (HuggingFace API)
        decoder_outputs = self.llm.model.decoder(input_ids=decoder_input_ids,
                                                 encoder_hidden_states=encoder_hidden,
                                                 encoder_attention_mask=enc_attention_mask,
                                                 return_dict=True)
        decoder_hidden = decoder_outputs.last_hidden_state  # (B, T_dec, d_model)

        logits = self.lm_head(decoder_hidden)  # (B, T_dec, vocab_size)

        loss = None
        if labels is not None:
            # labels: (B, T_dec) with ignore_index=-100 for padded tokens
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss