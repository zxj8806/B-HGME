import sys, os
sys.path.append('./')
import numpy as np
import torch
from .torch_sqrtm import MatrixSquareRoot

from torch.optim import Adam
import time
import warnings

from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
# from rich.progress import *
from rich.progress import track


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

torch_sqrtm = MatrixSquareRoot.apply

def isPSD(A, tol=1e-5):
    # Use torch.linalg.eigh for symmetric matrices
    E = torch.linalg.eigvalsh(A)  # Returns only the eigenvalues (real for symmetric matrices)
    
    # Check if all eigenvalues are greater than -tol (non-negative with tolerance)
    is_psd = torch.all(E > -tol)
    
    # Check if the matrix is symmetric (within tolerance)
    is_symmetric = torch.allclose(A, A.T, atol=tol)
    
    # Return whether the matrix is PSD and the minimum eigenvalue
    return is_psd & is_symmetric, torch.min(E)


def find_clostest_PSD(A):

    # Compute eigenvalues and eigenvectors using torch.linalg.eig
    E, U = torch.linalg.eig(A)  # E: eigenvalues, U: eigenvectors
    
    # Ensure eigenvalues are real (since A is symmetric in practice)
    E = torch.real(E)
    U = torch.real(U)
    
    # Set negative eigenvalues to a small positive value (e.g., 1e-3)
    E[E <= 0] = 1e-3
    
    # Reconstruct the PSD matrix
    S_diag = torch.diag(E)  # Create a diagonal matrix from eigenvalues
    A_psd = U @ S_diag @ U.T  # Reconstruct the matrix
    
    return A_psd

def weighted_kendall_tau(xs, ys, ws = None):

    n = xs.shape[0]
    if ws is None:
        ws = torch.ones(n)
    assert ys.shape[0] == n
    assert ws.shape[0] == n
    kt = 0
    norm = 0
    # mask array
    mx = (xs != 0)
    my = (ys != 0)
    # make signed_diffx[i,j] = np.signed(xs[i] - xs[j])
    signed_diffx = torch.sign(xs[:, None] - xs[None, :])
    # make signed_diffy[i,j] = np.signed(ys[i] - ys[j])
    signed_diffy = torch.sign(ys[:, None] - ys[None, :])
    # make w_mat[i,j] = mx[i] * my[i] * mx[j] * my[j] * ws[i] * ws[j]
    w_mat = mx[:, None] * my[:, None] * mx[None, :] * my[None, :] * ws[:, None] * ws[None, :]
    # make sure don't sum up diagonal value, where i == j
    w_mat.fill_diagonal_(0)
    kt = torch.sum(signed_diffx * signed_diffy * w_mat)/(torch.sum(w_mat) + 1e-12)
    del w_mat
    return kt
    
def est_cov(X, K_trun, weighted_kt = True):
    X = torch.FloatTensor(X)
    ntimes = X.shape[0]
    ngenes = X.shape[1]

    empir_cov = torch.zeros(ntimes, ngenes, ngenes)

    # building weighted covariance matrix, output is empir_cov of the shape (ntimes, ngenes, ngenes)
    for t in range(ntimes):
        weight = torch.FloatTensor(K_trun[t, :])
        if weighted_kt:
            weight = (weight > 0)
        # assert torch.sum(weight > 0) == 15

        # sample_mean = torch.sum(sample * (bin_weight/torch.sum(bin_weight))[:, None], dim = 0)
        for i in range(ngenes):
            for j in range(i, ngenes):
                if weighted_kt:
                    kt = weighted_kendall_tau(X[(weight > 0), i].squeeze(), X[(weight > 0), j].squeeze(), weight[weight > 0])
                    if i != j:
                        empir_cov[t, i, j] = torch.sin(np.pi/2 * kt)
                        # assert empir_cov[t, i, j] < 1
                    else:
                        empir_cov[t, i, j] = 1
                        
                else:
                    kt = weighted_kendall_tau(X[(weight > 0), i].squeeze(), X[(weight > 0), j].squeeze(), weight[weight > 0])
                    if i != j:
                        empir_cov[t, i, j] = torch.sin(np.pi/2 * kt) 
                        assert empir_cov[t, i, j] < 1
                    else:
                        empir_cov[t, i, j] = 1
        empir_cov[t,:,:] = empir_cov[t,:,:] + empir_cov[t,:,:].T - torch.diag(torch.diag(empir_cov[t,:,:]))
        # check positive definite
        Flag, min_eig = isPSD(empir_cov[t,:,:])
        # if not find the closest positive definite matrix
        if not Flag:
            # print("inferred empirical convariance matrix is not positive definite at the time point " + str(t) + ", min eig: " + str(min_eig))
            empir_cov[t,:,:] = find_clostest_PSD(empir_cov[t,:,:])
            Flag, min_eig = isPSD(empir_cov[t,:,:])
            # print("After correction, the minimum eigenvalue: " + str(min_eig))
            assert Flag

    return empir_cov


def _est_cov(X, K_trun, weighted_kt, t1, t2):
    X = torch.FloatTensor(X)
    ntimes = X.shape[0]
    ngenes = X.shape[1]
    empir_cov = torch.zeros(t2-t1, ngenes, ngenes)

    for t_idx, t in enumerate(np.arange(t1, t2)):
        weight = torch.FloatTensor(K_trun[t, :])
        if weighted_kt:
            weight = (weight > 0)

        for i in range(ngenes):
            for j in range(i, ngenes):
                if weighted_kt:
                    kt = weighted_kendall_tau(X[(weight > 0), i].squeeze(), X[(weight > 0), j].squeeze(), weight[weight > 0])
                    if i != j:
                        empir_cov[t_idx, i, j] = torch.sin(np.pi/2 * kt)
                    else:
                        empir_cov[t_idx, i, j] = 1
                        
                else:
                    kt = weighted_kendall_tau(X[(weight > 0), i].squeeze(), X[(weight > 0), j].squeeze(), weight[weight > 0])
                    if i != j:
                        empir_cov[t_idx, i, j] = torch.sin(np.pi/2 * kt) 
                        assert empir_cov[t_idx, i, j] < 1
                    else:
                        empir_cov[t_idx, i, j] = 1
    
        empir_cov[t_idx,:,:] = empir_cov[t_idx,:,:] + empir_cov[t_idx,:,:].T - torch.diag(torch.diag(empir_cov[t_idx,:,:]))
        # check positive definite
        Flag, min_eig = isPSD(empir_cov[t_idx,:,:])
        # if not find the closest positive definite matrix
        if not Flag:
            empir_cov[t_idx,:,:] = find_clostest_PSD(empir_cov[t_idx,:,:])
            # Flag, min_eig = isPSD(empir_cov[t_idx,:,:])
            # assert Flag

    return empir_cov

def est_cov_para(X, K_trun, weighted_kt = True, ncpus = 1):
    ntimes = X.shape[0]
    ngenes = X.shape[1]

    # building weighted covariance matrix, output is empir_cov of the shape (ntimes, ngenes, ngenes)
    settings = []
    start_time = np.arange(0, ntimes, ntimes//ncpus)
    end_time = np.append(start_time[1:], ntimes)

    for i, t in enumerate(start_time):
        settings.append((X.copy(), K_trun.copy(), weighted_kt, t, end_time[i]))
    print("calculating covariance matrices")
    # parallele
    pool = Pool(ncpus)
    empir_cov = pool.starmap(_est_cov, [setting for setting in settings])
    pool.close()
    pool.join()    
    empir_cov = np.concatenate(empir_cov, axis = 0)

    return empir_cov


class G_admm_minibatch():
    def __init__(self, X, K, TF = None, seed = 0, pre_cov = None, batchsize = None, device = device):
        super(G_admm_minibatch, self).__init__()
        # set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # shape (ntimes, nsamples, ngenes)
        self.X = torch.FloatTensor(X)
        self.ntimes, self.nsamples, self.ngenes = self.X.shape
        self.device = device
        
        # calculate batchsize
        if batchsize is None:
            self.batchsize = int(self.ntimes/10)
        else:
            self.batchsize = batchsize
        # calculate empirical covariance matrix
        if pre_cov is None:
            if X is None:
                raise ValueError('either X or pre_cov must be provided')
            # shape (ntimes, nsamples, ngenes)
            self.epir_mean = self.X.mean(dim = 1, keepdim = True)
            X = self.X - self.epir_mean
            # (ntimes * nsamples, ngenes, ngenes)
            self.empir_cov = torch.bmm(X.reshape((self.ntimes * self.nsamples, self.ngenes, 1)), X.reshape((self.ntimes * self.nsamples, 1, self.ngenes)))
            # (ntimes, ngenes, ngenes)
            self.empir_cov = torch.sum(self.empir_cov.reshape((self.ntimes, self.nsamples, self.ngenes, self.ngenes)), dim = 1)/(self.nsamples - 1)
        else:
            self.empir_cov = pre_cov

        # weight kernel function, shape (ntimes, ntimes)
        self.weights = torch.FloatTensor(K)
        # weighted average of empricial covariance matrix
        assert torch.all(torch.sum(self.weights,dim=1) - 1 < 1e-6)
        # memory consuming
        # self.w_empir_cov = torch.sum((self.weights[:,:,None,None]*self.empir_cov[None,:,:,:]),dim=1) 
        self.w_empir_cov = torch.stack([torch.sum(weight[:, None, None] * self.empir_cov, axis = 0) for weight in self.weights], dim = 0)
        assert self.w_empir_cov.shape[0] == self.ntimes
        assert self.w_empir_cov.shape[1] == self.ngenes
        assert self.w_empir_cov.shape[2] == self.ngenes
        # assert torch.equal(self.w_empir_cov, w_empir_cov)
        # store the result
        self.thetas = np.zeros((self.ntimes, self.ngenes, self.ngenes))
        
        # mask matrix (ntimes, ngenes, ngenes)
        if TF is not None:
            self.mask = torch.zeros(self.ngenes, self.ngenes)
            # mark probable interactions
            self.mask[TF, :] = 1
            self.mask[:, TF] = 1
            # remove the interactions between TFs
            self.mask[np.ix_(TF, TF)] = 0
            # element-wise reverse
            self.mask = 1 - self.mask
            self.mask = self.mask.expand(self.ntimes, self.ngenes, self.ngenes) 
        else:
            self.mask = torch.FloatTensor([0])  

    @staticmethod
    def neg_lkl_loss(thetas, S):

        # logdet works for batches of matrices, give a high dimensional data
        t1 = -1*torch.logdet(thetas)
        t2 = torch.stack([torch.trace(mat) for mat in torch.bmm(S, thetas)])
        return t1 + t2


    def train(self, max_iters = 50, n_intervals = 1, lamb = 2.1e-4, alpha = 2, rho = 1.7, beta = 0, theta_init_offset = 0.1):
        n_batches = int(np.ceil(self.ntimes/self.batchsize))
        # pbar = Progress()
        # task = pbar.add_task("[cyan]1. Inferring GRNs with CeSpGRN", total=n_batches)
        for batch in track(range(n_batches), description="[cyan]1. Inferring GRNs with CeSpGRN"):
            # print("start running batch " + str(batch))
            start_time = time.time()
            # select a minibatch, and load to cuda
            start_idx = batch * self.batchsize
            if batch < n_batches - 1:
                end_idx = (batch + 1) * self.batchsize
                w_empir_cov = self.w_empir_cov[start_idx:end_idx, :, :].to(self.device)
                if self.mask.shape[0] == self.ntimes:
                    mask = self.mask[start_idx:end_idx, :, :].to(self.device)
                else:
                    mask = self.mask.to(self.device)
            else:
                w_empir_cov = self.w_empir_cov[start_idx:, :, :].to(self.device)
                if self.mask.shape[0] == self.ntimes:
                    mask = self.mask[start_idx:, :, :].to(self.device)
                else:
                    mask = self.mask.to(self.device)
            # initialize mini-batch, Z of the shape (batch_size, ngenes, ngenes)
            Z = torch.diag_embed(1/(torch.diagonal(w_empir_cov, offset=0, dim1=-2, dim2=-1) + theta_init_offset))
            # make Z positive definite matrix
            ll = torch.cholesky(Z)
            Z = torch.matmul(ll, ll.transpose(-1, -2))
            U = torch.zeros(Z.shape).to(self.device)
            I = torch.eye(self.ngenes).expand(Z.shape).to(self.device)

            it = 0
            its = []
            losses = []
            losses1 = []
            losses2 = []
            losses3 = []
            # hyper-parameter for batches
            if rho is None:
                updating_rho = True
                # rho of the shape (ntimes, 1, 1)
                b_rho = torch.ones((Z.shape[0], 1, 1)).to(self.device) * 1.7
            else:
                b_rho = torch.FloatTensor([rho] * Z.shape[0])[:, None, None].to(self.device)
                updating_rho = False
            b_alpha = alpha 
            b_beta = beta
            b_lamb = lamb
            while(it < max_iters): 
                # Primal 
                Y = U - Z + w_empir_cov/b_rho    # (ntimes, ngenes, ngenes)
                thetas = - 0.5 * Y + torch.stack([torch_sqrtm(mat) for mat in (torch.transpose(Y,1,2) @ Y * 0.25 + I/b_rho)])
                Z_pre = Z.detach().clone()
                # over-relaxation
                thetas = b_alpha * thetas + (1 - b_alpha) * Z_pre            
                Z = torch.sign(thetas + U) * torch.max((b_rho * (thetas + U).abs() - b_lamb)/(b_rho + b_beta * mask), torch.Tensor([0]).to(self.device))

                # Dual
                U = U + thetas - Z

                # calculate residual
                # primal_residual and dual_residual of the shape (ntimes, 1, 1)
                primal_residual = torch.sqrt((thetas - Z).pow(2).sum(1).sum(1))
                dual_residual = b_rho.squeeze() * torch.sqrt((Z - Z_pre).pow(2).sum(1).sum(1))

                # updating rho, rho should be of shape (ntimes, 1, 1)
                if updating_rho:
                    mask_inc = (primal_residual > 10 * dual_residual)
                    b_rho[mask_inc, :, :] = b_rho[mask_inc, :, :] * 2
                    mask_dec = (dual_residual > 10 * primal_residual)
                    b_rho[mask_dec, :, :] = b_rho[mask_dec, :, :] / 2
                
                # print(rho.squeeze())
                # free-up memory
                del Z_pre
                
                # Stopping criteria
                if (it + 1) % n_intervals == 0:
                    # calculate sum of all duality gap
                    # loss = self.neg_lkl_loss(thetas, w_empir_cov).sum() + b_lamb * Z.abs().sum() + b_beta * (self.mask * Z).pow(2).sum()
                    # primal_val = loss  + rho/2 * (thetas - Z).pow(2).sum()
                    # dual_val = loss + rho/2 * (thetas - Z + U).pow(2).sum() - rho/2 * U.pow(2).sum()
                    # duality_gap = primal_val - dual_val

                    # simplify min of all duality gap
                    duality_gap = b_rho.squeeze() * torch.stack([torch.trace(mat) for mat in torch.bmm(U.permute(0,2,1), Z - thetas)])
                    duality_gap = duality_gap.abs()

                    # loss function
                    loss1 = self.neg_lkl_loss(Z, w_empir_cov).sum()
                    loss2 = Z.abs().sum()
                    loss3 = (mask * Z).pow(2).sum()
                    loss = loss1 + b_lamb * loss2 + b_beta * loss3
                    losses.append(loss.item())
                    losses1.append(loss1.item())
                    losses2.append(loss2.item())
                    losses3.append(loss3.item())
                    its.append(it)
                    
                    # if duality_gap < 1e-8:
                    #     break
                    primal_eps = 1e-3 # 1e-6
                    dual_eps = 1e-3 # 1e-6
                    if (primal_residual.max() < primal_eps) and (dual_residual.max() < dual_eps):
                        break                
                it += 1
            
            loss1 = self.neg_lkl_loss(Z, w_empir_cov).sum()
            loss2 = Z.abs().sum()
            loss3 = (mask * Z).pow(2).sum()

            # store values
            if batch < n_batches - 1:
                self.thetas[start_idx:end_idx] = Z.detach().cpu().numpy()
            else:
                self.thetas[start_idx:] = Z.detach().cpu().numpy()
            del thetas, U, I, Y, ll, Z

            # print("Finished running batch {:d}, running time: {:.2f} sec".format(batch, time.time() - start_time))

            # remove target-target if beta > 0
            if beta > 0:
                for t, theta in enumerate(self.thetas):
                    self.thetas[t] = theta * (1 - self.mask[0].detach().cpu().numpy())
                for t, theta in enumerate(self.thetas):
                    assert np.sum(theta * self.mask[0].detach().cpu().numpy()) == 0            

        return self.thetas


def _construct_one_G(theta):
    ngenes = theta.shape[0]
    G = np.zeros_like(theta)
    for i in range(ngenes - 1):
        for j in range(i+1, ngenes):
            # inverse transform the precision matrix back to conditional covariance matrix, element 0,1 is the conditional covariance
            G[i, j] = np.linalg.pinv(theta[np.ix_([i,j], [i,j])])[0,1]
            G[j, i] = G[i, j]
    return G[None,:,:]

def construct_weighted_G(thetas, ncpus = 8):
    print("Construct conditional covariance matrix...")
    ntimes = thetas.shape[0]
    ngenes = thetas.shape[1]
    pool = Pool(ncpus)

    Gs = pool.map(_construct_one_G, [theta for theta in thetas])
    pool.close()
    pool.join()    
    Gs = np.concatenate(Gs, axis = 0)
    print("Done.")
    return Gs
