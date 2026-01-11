import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(CNN, self).__init__()

        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, [4, 2], 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1

        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(int(m[0])):
                    x = int(m[1])
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x,))
                    in_channels = x
                    cnt += 1

    def forward(self, x, reduce='max'):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        return y

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


class CTCtopC(nn.Module):
    def __init__(self, input_size, nclasses, dropout=0.0):
        super(CTCtopC, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.cnn_top = nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):
    
        x = self.dropout(x)

        y = self.cnn_top(x)
        y = y.permute(2, 3, 0, 1)[0]
        return y


class CTCtopR(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        super(CTCtopR, self).__init__()

        hidden, num_layers = rnn_cfg

        if rnn_type == 'gru':
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        elif rnn_type == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        else:
            print('problem! - no such rnn type is defined')
            exit()
        
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))

    def forward(self, x):

        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)

        return y

class CTCtopB(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        super(CTCtopB, self).__init__()

        hidden, num_layers = rnn_cfg

        if rnn_type == 'gru':
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        elif rnn_type == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        else:
            print('problem! - no such rnn type is defined')
            exit()
        
        self.fnl = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Sequential(nn.Dropout(.5), 
                                 nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )

    def forward(self, x):

        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]

        y = self.fnl(y)

        if self.training:
            return y, self.cnn(x).permute(2, 3, 0, 1)[0]
        else:
            return y, self.cnn(x).permute(2, 3, 0, 1)[0]


class ViTRGTSBackbone(nn.Module):
    """
    ViT-style backbone with register tokens, inspired by:
    - 'Vision Transformers Need Registers'
    - kyegomez/Vit-RGTS

    Input:  x : [B, 1, H, W]  (grayscale line image)
    Output: seq_tokens : [T, B, D]  (patch tokens for CTC)
            reg_tokens : [B, R, D]  (register tokens for analysis later)
            grid_size  : (Hp, Wp)   (patch grid, for spatial mapping of tokens)
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_registers: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.max_seq_len = max_seq_len

        # Patch embedding: from [B, 1, H, W] -> [B, D, Hp, Wp]
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Register tokens (learned, store global info)
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_registers, embed_dim)
        )

        # Learned positional embeddings for (registers + patches)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )

        self.emb_dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # [B, S, D]
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        # patch_embed uses Kaiming by default; you can tweak if you want stronger init

    def forward(self, x):
        """
        x: [B, 1, H, W]
        returns:
            seq_tokens: [T, B, D]   (for CTC head)
            reg_tokens: [B, R, D]   (for analysis)
            grid_size : (Hp, Wp)
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected grayscale input (C=1), got C={C}"

        # Patchify
        x = self.patch_embed(x)               # [B, D, Hp, Wp]
        B, D, Hp, Wp = x.shape
        num_patches = Hp * Wp

        patch_tokens = x.flatten(2).transpose(1, 2)  # [B, Np, D]

        # Prepare register tokens
        reg_tokens = self.register_tokens.expand(B, -1, -1)  # [B, R, D]

        # Concatenate registers + patches -> [B, S, D]
        tokens = torch.cat([reg_tokens, patch_tokens], dim=1)  # S = R + Np

        S = tokens.size(1)
        if S > self.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len={self.max_seq_len}. "
                "Increase vit_max_seq_len in your config."
            )

        # Add positional embeddings and dropout
        pos = self.pos_embed[:, :S, :]  # [1, S, D]
        tokens = tokens + pos
        tokens = self.emb_dropout(tokens)

        # Transformer encoder
        encoded = self.encoder(tokens)         # [B, S, D]

        # Split back into registers and patch tokens
        reg_out = encoded[:, :self.num_registers, :]      # [B, R, D]
        patch_out = encoded[:, self.num_registers:, :]    # [B, Np, D]

        # For CTC, we want [T, B, D] (time-major)
        seq_tokens = patch_out.transpose(0, 1)            # [Np, B, D]

        return seq_tokens, reg_out, (Hp, Wp)
    
    @torch.no_grad()
    def forward_explain(self, x):
        """
        Explainability forward pass.
        Returns:
            seq_tokens:   [T, B, D]           - patch token sequence (CTC input)
            reg_tokens:   [B, R, D]           - register embeddings
            attn_maps:    List[L] of [B, H, S, S]  - per-layer, per-head attention
            token_norms:  [B, S]              - L2 norm per token at final layer
            grid_size:    (Hp, Wp)
        """
        B, C, H, W = x.shape

        # === PATCH EMBEDDING ===================================================
        x = self.patch_embed(x)               # [B, D, Hp, Wp]
        B, D, Hp, Wp = x.shape
        patch_tokens = x.flatten(2).transpose(1, 2)  # [B, Np, D]
        num_patches = Hp * Wp

        reg_tokens = self.register_tokens.expand(B, -1, -1)  # [B, R, D]

        tokens = torch.cat([reg_tokens, patch_tokens], dim=1)   # [B, S, D]
        S = tokens.size(1)

        pos = self.pos_embed[:, :S, :]
        tokens = tokens + pos
        tokens = self.emb_dropout(tokens)

        # === INITIALIZE EXPLAINABILITY BUFFERS =================================
        attn_maps = []   # list of per-layer attention
        token_norms = None

        # === MANUAL TRANSFORMER ENCODER STACK ==================================
        x_tokens = tokens
        for layer in self.encoder.layers:
            # Access internal attention module
            attn_block = layer.self_attn

            # ---- compute attention manually ----
            # (batch_first=True)
            q = attn_block.q_proj(x_tokens)
            k = attn_block.k_proj(x_tokens)
            v = attn_block.v_proj(x_tokens)

            B_, S_, D_ = q.shape
            H = attn_block.num_heads
            d_head = D_ // H

            q = q.view(B_, S_, H, d_head).transpose(1, 2)   # [B, H, S, Dh]
            k = k.view(B_, S_, H, d_head).transpose(1, 2)
            v = v.view(B_, S_, H, d_head).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_head)
            attn = torch.softmax(attn, dim=-1)              # [B, H, S, S]

            attn_maps.append(attn.cpu())

            # usual attention output
            out = torch.matmul(attn, v)                     # [B, H, S, Dh]
            out = out.transpose(1, 2).contiguous().view(B_, S_, D_)
            out = attn_block.out_proj(out)

            # full layer: norm-first encoder
            x_tokens = x_tokens + out
            x_tokens = x_tokens + layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x_tokens)))))

        # === SPLIT REGISTERS & PATCH TOKENS =====================================
        reg_out = x_tokens[:, :self.num_registers, :]            # [B, R, D]
        patch_out = x_tokens[:, self.num_registers:, :]          # [B, Np, D]

        # === TOKEN NORMS (FINAL LAYER) ==========================================
        token_norms = x_tokens.norm(dim=-1).cpu()                # [B, S]

        # === FINAL SHAPE FOR CTC ===============================================
        seq_tokens = patch_out.transpose(0, 1)                   # [T, B, D]

        return seq_tokens, reg_out, attn_maps, token_norms, (Hp, Wp)



class HTRNet(nn.Module):
    """
    Unified entry point for:
      - CNN+RNN (original Best Practices model): arch_cfg.type == 'cnn_rnn'
      - ViT+Registers backbone (this work):     arch_cfg.type == 'vit_rgts'

    Forward always returns:
        - logits: [T, B, nclasses]   (CTC-ready)
    """

    def __init__(self, arch_cfg, nclasses):
        super(HTRNet, self).__init__()

        self.arch_type = getattr(arch_cfg, "type", "cnn_rnn")

        # (Optional) STN hook – still not implemented
        if getattr(arch_cfg, "stn", False):
            raise NotImplementedError(
                "Spatial Transformer Networks not implemented - "
                "you can plug your own STN here if needed."
            )
        self.stn = None

        # ------------------------------------------------------------------
        # 1) CNN + RNN path  (original Best Practices HTRNet)
        # ------------------------------------------------------------------
        if self.arch_type == "cnn_rnn":
            cnn_cfg = arch_cfg.cnn_cfg
            self.features = CNN(cnn_cfg, flattening=arch_cfg.flattening)

            # Hidden size after CNN flattening
            if arch_cfg.flattening in ["maxpool", "avgpool"]:
                hidden = cnn_cfg[-1][-1]          # last channel
            elif arch_cfg.flattening == "concat":
                # In original code, k is the pooling height in the last layer
                # If you changed that, expose it in arch_cfg and use it here.
                k = getattr(arch_cfg, "k", 4)
                hidden = cnn_cfg[-1][-1] * k
            else:
                raise ValueError(
                    f"Unknown flattening mode: {arch_cfg.flattening}"
                )

            head = arch_cfg.head_type
            if head == "cnn":
                self.top = CTCtopC(hidden, nclasses)
            elif head == "rnn":
                self.top = CTCtopR(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
            elif head == "both":
                self.top = CTCtopB(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
            else:
                raise ValueError(f"Unknown head_type: {head}")

               # ------------------------------------------------------------------
        # 2) ViT + Registers path  (Vision Transformers Need Registers)
        # ------------------------------------------------------------------
        elif self.arch_type == "vit_rgts":
            # Read ViT hyper-parameters from baseline_vit_rgts.yaml
            # and provide sensible fallbacks.

            image_height = getattr(arch_cfg, "image_height", 128)
            image_width  = getattr(arch_cfg, "image_width", 1024)

            # Allow either (patch_height/patch_width) or a single patch_size
            patch_height = getattr(arch_cfg, "patch_height", getattr(arch_cfg, "patch_size", 16))
            patch_width  = getattr(arch_cfg, "patch_width", patch_height)

            dim    = getattr(arch_cfg, "dim", getattr(arch_cfg, "vit_embed_dim", 256))
            depth  = getattr(arch_cfg, "depth", getattr(arch_cfg, "vit_depth", 6))
            heads  = getattr(arch_cfg, "heads", getattr(arch_cfg, "vit_num_heads", 8))
            mlp_dim = getattr(arch_cfg, "mlp_dim", int(dim * 4))

            # Convert mlp_dim into a ratio if needed
            mlp_ratio = getattr(arch_cfg, "vit_mlp_ratio", float(mlp_dim) / float(dim))

            num_registers = getattr(arch_cfg, "num_registers", 4)

            dropout = getattr(arch_cfg, "dropout", getattr(arch_cfg, "vit_dropout", 0.1))
            emb_dropout = getattr(arch_cfg, "emb_dropout", getattr(arch_cfg, "vit_emb_dropout", 0.1))

            # Compute a safe max_seq_len if not specified:
            #   R registers + Hp*Wp patches + small margin
            Hp = image_height // patch_height
            Wp = image_width  // patch_width
            seq_len_est = num_registers + Hp * Wp + 8

            max_seq_len = getattr(arch_cfg, "vit_max_seq_len", seq_len_est)

            self.backbone = ViTRGTSBackbone(
                image_size=image_height,         # currently unused internally, but kept for clarity
                patch_size=patch_height,         # we assume square patches; you can extend if needed
                embed_dim=dim,
                depth=depth,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                num_registers=num_registers,
                dropout=dropout,
                emb_dropout=emb_dropout,
                max_seq_len=max_seq_len,
            )

            hidden = self.backbone.embed_dim

            # You can still choose to stack an RNN on top of the ViT tokens,
            # or go pure-transformer with a linear CTC head.
            head = getattr(arch_cfg, "head_type", "rnn")

            if head == "cnn":
                # 'cnn' in this context = linear over ViT sequence (no RNN)
                self.top = CTCtopC(hidden, nclasses)
            elif head == "both":
                self.top = CTCtopB(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )
            else:
                # default: ViT -> BiRNN -> CTC
                self.top = CTCtopR(
                    hidden,
                    (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
                    nclasses,
                    rnn_type=arch_cfg.rnn_type,
                )

        else:
            raise ValueError(f"Unknown architecture type: {self.arch_type}")

    def forward(self, x):
        """
        x: [B, 1, H, W]
        returns:
            logits_ctc: [T, B, nclasses]
        """

        # Optional spatial transformer
        if self.stn is not None:
            x = self.stn(x)

        if self.arch_type == "cnn_rnn":
            # CNN features already return [T, B, C]
            seq = self.features(x)      # [T, B, C]
            logits = self.top(seq)      # [T, B, nclasses]
            return logits

        elif self.arch_type == "vit_rgts":
            # ViT backbone returns [T, B, D]
            seq_tokens, reg_tokens, grid_size = self.backbone(x)   # [T, B, D]

            # Adapt to the 4D format expected by CTCtop{C,R,B}:
            # original heads expect [B, C, H, W] with H=1, W=T
            # ViT output is [T, B, D] -> [B, D, 1, T]
            seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)      # [B, D, 1, T]

            logits = self.top(seq_4d)                              # [T, B, nclasses] or (T,B,nclasses,aux)
            return logits

            raise ValueError(f"Unknown architecture type at forward: {self.arch_type}")

    @torch.no_grad()
    def forward_explain(self, x):
        """
        Returns logits + attention + registers + norms for analysis
        """
        if self.arch_type != "vit_rgts":
            raise ValueError("forward_explain only supported for vit_rgts")

        seq_tokens, reg_tokens, attn_maps, token_norms, grid = self.backbone.forward_explain(x)

        # reshape seq_tokens to 4D for the top head
        seq_4d = seq_tokens.permute(1, 2, 0).unsqueeze(2)  # [B, D, 1, T]
        logits = self.top(seq_4d)  # [T, B, C]

        return logits, reg_tokens, attn_maps, token_norms, grid

