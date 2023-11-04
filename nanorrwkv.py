import torch
import torch.nn.functional as F

file = "1.5v5.pth"
# download 1b5 from https://huggingface.co/BlinkDL/rwkv-5-world/blob/main/RWKV-5-World-1B5-v2-20231025-ctx4096.pth

dims = 2048
dims_fnn = int(2048*3.5)
dims_att = 2048
n_head = 32
n_headsize = 64
vocab = 2**16
layers = 24

class TimeShift(torch.nn.Module):
    def forward(self, x, state):
        xapp = torch.cat([state, x], dim=-2)
        return xapp[:,:-1,:], xapp[:,-1:]
    

class Feed_Forward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_shift = TimeShift()

        self.time_mix_k = torch.nn.Parameter(torch.zeros(1,1,dims))
        self.time_mix_r = torch.nn.Parameter(torch.zeros(1,1,dims))
    
        self.key = torch.nn.Linear(dims, dims_fnn, bias=False)
        self.receptance = torch.nn.Linear(dims, dims, bias=False)
        self.value = torch.nn.Linear(dims_fnn, dims, bias=False)

    
    def forward(self, x, state):
        
        xx, state = self.time_shift(x,state)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, state
    

class Att(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ddd = torch.ones(1, 1, dims)
        self.time_mix_k = torch.nn.Parameter(ddd)
        self.time_mix_v = torch.nn.Parameter(ddd)
        self.time_mix_r = torch.nn.Parameter(ddd)
        self.time_mix_g = torch.nn.Parameter(ddd)
        self.time_decay = torch.nn.Parameter(torch.ones(n_head, n_headsize))
        self.time_faaaa = torch.nn.Parameter(torch.ones(n_head, n_headsize))

        self.gate =torch.nn.Linear(dims, dims_att, bias=False)
        self.time_shift = TimeShift()
        self.receptance =torch.nn.Linear(dims, dims_att, bias=False)
        self.key =torch.nn.Linear(dims, dims_att, bias=False)
        self.value =torch.nn.Linear(dims, dims_att, bias=False)
        self.output =torch.nn.Linear(dims_att, dims, bias=False)

        self.ln_x = torch.nn.GroupNorm(n_head, dims_att)

    def forward(self, x, attstate, wkvstate):
        B,T,C = x.size()
        
        xx, attstate = self.time_shift(x, attstate) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr).view(B, T, n_head, 1, -1)
        k = self.key(xk).view(B, T, n_head, -1, 1)
        v = self.value(xv).view(B, T, n_head, 1, -1)
        g = F.silu(self.gate(xg))
        
        at = k @ v
       
        u = self.time_faaaa.view(1,1,n_head, 1, -1)
        out = (u * r ) @ at
        w = self.time_decay.double().exp().neg().exp().reshape(1,n_head,-1,1).to(x.dtype)
        
        for t in range(T):
            
            out[:,t] += r[:,t] @ wkvstate
            wkvstate *= w
            wkvstate += at[:,t]
            

        x = out.view(-1, C)
        
        x = self.ln_x(x / 8).view(B, T, C)
        x = self.output(x * g)
        return x, attstate, wkvstate
    
    
class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        
        self.ln1 = torch.nn.LayerNorm(dims)
        self.ln2 = torch.nn.LayerNorm(dims)

        if self.layer_id == 0:
            self.ln0 = torch.nn.LayerNorm(dims)

        self.att = Att()
        self.ffn = Feed_Forward()
    
    def forward(self, inp):
        x, ffnstate,attstate,wkvstate = inp
        if self.layer_id == 0:
            x = self.ln0(x)

        att_out, attstate[self.layer_id], wkvstate[self.layer_id] = self.att(self.ln1(x),attstate[self.layer_id],wkvstate[self.layer_id])
        x = x + att_out
        
        ffn_out, ffnstate[self.layer_id] = self.ffn(self.ln2(x),ffnstate[self.layer_id])
        x = x + ffn_out

        return x, ffnstate, attstate, wkvstate

class RWKV(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dims)
        self.blocks = torch.nn.Sequential(*[Block(i) for i in range(layers)])
        self.ln_out = torch.nn.LayerNorm(dims)
        self.head = torch.nn.Linear(dims, vocab, bias=False)

    def forward(self,x, ffnstate, attstate, wkvstate):

        x = self.emb(x)
        x, ffnstate, attstate, wkvstate = self.blocks((x, ffnstate, attstate, wkvstate))
        x = self.ln_out(x)
        x = self.head(x)

        return x, ffnstate, attstate, wkvstate

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[max(i,1)], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

tokenizer = RWKV_TOKENIZER("vocab.txt")

model = RWKV()

model.load_state_dict(torch.load(file, map_location="cpu"))
model = model.float()

testtext = "the best language model is RWKV because"
print(testtext)

toktext = tokenizer.encode(testtext)

ffnstate, attstate, wkvstate = \
    torch.zeros(layers,1,1,dims), \
    torch.zeros(layers,1,1,dims), \
    torch.zeros(layers, 1,n_head,n_headsize,n_headsize)

toks = torch.tensor(toktext).reshape(1,-1)

probs, ffnstate, attstate, wkvstate = model.forward(toks,ffnstate, attstate, wkvstate)

for i in range(1000):
    lasttoken = torch.argmax(probs[0,-1]).reshape(1,1)
    probs, ffnstate, attstate, wkvstate = model.forward(lasttoken,ffnstate, attstate, wkvstate)
    try:
        print(tokenizer.decode([lasttoken.reshape(-1).item()]), flush=True, end="")
    except:
        pass

