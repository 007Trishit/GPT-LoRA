from GPTmodel import *


@dataclass
class LoRAGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    LoRA_rank: int = 2


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALinear, self).__init__()
        self.L = nn.Linear(in_features, rank)
        self.R = nn.Linear(rank, out_features)
        # set the weights and bias to zero
        self.L.weight.data.fill_(0)
        self.L.bias.data.fill_(0)
        self.R.weight.data.fill_(0)
        self.R.bias.data.fill_(0)

    def forward(self, x):
        return self.R(self.L(x))


class LoRACausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(
                torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size))
        # LoRA modifications:
        # remove gradients from the linear layers
        for layer in [self.c_attn, self.c_proj]:
            for param in layer.parameters():
                param.requires_grad = False
        # add LoRA linear layers
        self.aux_c_attn = LoRALinear(config.n_embd, 3 * config.n_embd, rank=config.LoRA_rank)
        self.aux_c_proj = LoRALinear(config.n_embd, config.n_embd, rank=config.LoRA_rank)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        c_attn_outs = self.c_attn(x)
        aux_c_attn_outs = self.aux_c_attn(x) # LoRA alternate path
        q, k, v  = (c_attn_outs + aux_c_attn_outs).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        c_proj_outs = self.c_proj(y)
        aux_c_proj_outs = self.aux_c_proj(y) # LoRA alternate path
        y = self.resid_dropout(c_proj_outs + aux_c_proj_outs)
        return y


class LoRAMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        # LoRA modifications:
        # remove gradients from the linear layers
        for layer in [self.c_fc, self.c_proj]:
            for param in layer.parameters():
                param.requires_grad = False
        # add LoRA linear layers
        self.aux_c_fc = LoRALinear(config.n_embd, 4 * config.n_embd, rank=config.LoRA_rank)
        self.aux_c_proj = LoRALinear(4 * config.n_embd, config.n_embd, rank=config.LoRA_rank)

    def forward(self, x):
        aux_x = self.aux_c_fc(x)
        x = self.c_fc(x)
        x = x + aux_x
        x = self.gelu(x)
        aux_x = self.aux_c_proj(x)
        x = self.c_proj(x)
        x = x + aux_x
        x = self.dropout(x)
        return x


class LoRABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = LoRACausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = LoRAMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LoRAGPT(nn.Module):
    def __init__(self):
        super(LoRAGPT, self).__init__()
        # from transformers import GPT2LMHeadModel
        from transformers import GPT2ForSequenceClassification
        print("loading weights from pretrained gpt2 model")

        model_type = 'gpt2-medium'
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M (1.3B) params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        config_args['LoRA_rank'] = 4 # decomposition rank for LoRA

        self.config = LoRAGPTConfig(**config_args)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([LoRABlock(self.config) for _ in range(self.config.n_layer)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))
        # self.lm_head = nn.Linear(
        #     self.config.n_embd, self.config.vocab_size, bias=False)
        self.score = nn.Linear(self.config.n_embd, 2, bias=False)
        # self.transformer.wte.weight = self.lm_head.weight

        # Remove gradients from the embedding layers
        for layer in [self.transformer.wte, self.transformer.wpe]:
            for param in layer.parameters():
                param.requires_grad = False

        sd = self.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2ForSequenceClassification.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # # print all the parameters, their names, and whether they are trainable
        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(f"{n}")
        
        # print the number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params / 1e6:.2f}M")
        # print the number of trainable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
        # calculate the reduction in parameters
        reduction = 100 * (total_params - num_params) / total_params
        print(f"Reduction: {reduction:.2f}%")
        
    
    def forward(self, idx, mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # if mask is provided, find the indices of the last tokens in each sequence
        if mask is not None:
            assert mask.size() == idx.size(), "Mask size must match input size"
            eos_idxs = mask.sum(1) - 1

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if mask is not None:
            # if mask is provided, only return the logits for the last token in each sequence
            logits = self.score(x[torch.arange(b, device=device), eos_idxs])
        else:
            logits = self.score(x[:, -1, :])

        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def save_trainable_params(self, path):
        trainable_params =\
            list(filter(lambda p: p.requires_grad, self.parameters()))
        torch.save(trainable_params, path)
    
    def load_trainable_params(self, path):
        trainable_params = torch.load(path)
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = trainable_params.pop(0)
