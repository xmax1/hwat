
from functools import reduce
import torch 
import torch.nn as nn

def logabssumdet(xs):
        
        dets = [x.reshape(-1) for x in xs if x.shape[-1] == 1]						# in case n_u or n_d=1, no need to compute determinant
        dets = reduce(lambda a,b: a*b, dets) if len(dets)>0 else 1.					# take product of these cases
        maxlogdet = 0.																# initialised for sumlogexp trick (for stability)
        det = dets																	# if both cases satisfy n_u or n_d=1, this is the determinant
        
        slogdets = [torch.linalg.slogdet(x) for x in xs if x.shape[-1]>1] 			# otherwise take slogdet
        if len(slogdets)>0: 
            sign_in, logdet = reduce(lambda a,b: (a[0]*b[0], a[1]+b[1]), slogdets)  # take product of n_u or n_d!=1 cases
            maxlogdet = torch.max(logdet)											# adjusted for new inputs
            det = sign_in * dets * torch.exp(logdet-maxlogdet)						# product of all these things is determinant
        
        psi_ish = torch.sum(det)
        sgn_psi = torch.sign(psi_ish)
        log_psi = torch.log(torch.abs(psi_ish)) + maxlogdet
        return log_psi, sgn_psi


class FermiNetTorch(nn.Module):
    def __init__(self, n_e, n_u, n_d, n_det, n_fb, n_pv, n_sv, a, with_sign=False):
        super(FermiNetTorch, self).__init__()
        self.n_e = n_e                  # number of electrons
        self.n_u = n_u                  # number of up electrons
        self.n_d = n_d                  # number of down electrons
        self.n_det = n_det              # number of determinants
        self.n_fb = n_fb                # number of feedforward blocks
        self.n_pv = n_pv                # latent dimension for 2-electron
        self.n_sv = n_sv                # latent dimension for 1-electron
        self.a = a                      # nuclei positions
        self.with_sign = with_sign      # return sign of wavefunction

        self.n1 = [4*self.a.shape[0]] + [self.n_sv]*self.n_fb
        self.n2 = [4] + [self.n_pv]*(self.n_fb - 1)
        assert (len(self.n1) == self.n_fb+1) and (len(self.n2) == self.n_fb)
        self.Vs = nn.ModuleList([nn.Linear(3*self.n1[i]+2*self.n2[i], self.n1[i+1]) for i in range(self.n_fb)])
        self.Ws = nn.ModuleList([nn.Linear(self.n2[i], self.n2[i+1]) for i in range(self.n_fb-1)])

        self.V_half_u = nn.Linear(self.n_sv, self.n_sv // 2)
        self.V_half_d = nn.Linear(self.n_sv, self.n_sv // 2)

        self.wu = nn.Linear(self.n_sv // 2, self.n_u)
        self.wd = nn.Linear(self.n_sv // 2, self.n_d)

        # TODO: Multideterminant. If n_det > 1 we should map to n_det*n_u (and n_det*n_d) instead,
        #  and then split these outputs in chunks of n_u (n_d)
        # TODO: implement layers for sigma and pi

    def forward(self, r: torch.Tensor):
        """
        Batch dimension is not yet supported.
        """

        if len(r.shape) == 1:
            r = r.reshape(self.n_e, 3) # (n_e, 3)

        eye = torch.eye(self.n_e, device=r.device).unsqueeze(-1)

        ra = r[:, None, :] - self.a[None, :, :] # (n_e, n_a, 3)
        ra_len = torch.norm(ra, dim=-1, keepdim=True) # (n_e, n_a, 1)

        rr = r[None, :, :] - r[:, None, :] # (n_e, n_e, 1)
        rr_len = torch.norm(rr+eye, dim=-1, keepdim=True) * (torch.ones((self.n_e, self.n_e, 1))-eye) # (n_e, n_e, 1) 
        # TODO: Just remove '+eye' from above, it's unnecessary

        s_v = torch.cat([ra, ra_len], dim=-1).reshape(self.n_e, -1) # (n_e, n_a*4)
        p_v = torch.cat([rr, rr_len], dim=-1) # (n_e, n_e, 4)

        for l, (V, W) in enumerate(zip(self.Vs, self.Ws)):
            sfb_v = [torch.tile(_v.mean(dim=0)[None, :], (self.n_e, 1)) for _v in torch.split(s_v, 2, dim=0)]
            pfb_v = [_v.mean(dim=0) for _v in torch.split(p_v, self.n_u, dim=0)]
            
            s_v = torch.cat(sfb_v+pfb_v+[s_v,], dim=-1) # s_v = torch.cat((s_v, sfb_v[0], sfb_v[1], pfb_v[0], pfb_v[0]), dim=-1)
            s_v = torch.tanh(V(s_v)) + (s_v if (s_v.shape[-1]==self.n_sv) else 0.)
            
            if not (l == (self.n_fb-1)):
                p_v = torch.tanh(W(p_v)) + (p_v if (p_v.shape[-1]==self.n_pv) else 0.)
        
        s_u, s_d = torch.split(s_v, self.n_u, dim=0)

        s_u = torch.tanh(self.V_half_u(s_u)) # spin dependent size reduction
        s_d = torch.tanh(self.V_half_d(s_d))

        s_wu = self.wu(s_u) # map to phi orbitals
        s_wd = self.wd(s_d)

        assert s_wd.shape == (self.n_d, self.n_d)

        ra_u, ra_d = torch.split(ra, self.n_u, dim=0)

        # TODO: implement sigma = nn.Linear() before this
        exp_u = torch.norm(ra_u, dim=-1, keepdim=True)
        exp_d = torch.norm(ra_d, dim=-1, keepdim=True)

        assert exp_d.shape == (self.n_d, self.a.shape[0], 1)

        # TODO: implement pi = nn.Linear() before this
        orb_u = (s_wu * (torch.exp(-exp_u).sum(axis=1)))[None, :, :]
        orb_d = (s_wd * (torch.exp(-exp_d).sum(axis=1)))[None, :, :]

        assert orb_u.shape == (1, self.n_u, self.n_u)

        log_psi, sgn = logabssumdet([orb_u, orb_d])

        if self.with_sign:
            return log_psi, sgn
        else:
            return log_psi.squeeze()