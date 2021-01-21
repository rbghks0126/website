#!/usr/bin/env python
import numpy as np
import scipy.stats

import sys

from sklearn.base import BaseEstimator, TransformerMixin

EPS = np.spacing(1)


class SSMF_BP_NMF(BaseEstimator, TransformerMixin):
    '''
    Stochastic structured mean-field variational inference for Beta process
    Poisson NMF
    '''

    def __init__(self, n_components=500, max_iter=50, burn_in=1000,
                 cutoff=1e-3, smoothness=100, random_state=None,
                 verbose=False, **kwargs):
        self.n_components = n_components
        self.max_iter = max_iter
        self.burn_in = burn_in
        self.cutoff = cutoff
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        # hyperparameters for components
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))

        # hyperparameters for activation
        self.c = float(kwargs.get('c', 0.1))
        self.d = float(kwargs.get('d', 0.1))

        # hyperparameters for sparsity on truncated beta process
        self.a0_H = float(kwargs.get('a0_H', 1.))
        self.b0_H = float(kwargs.get('b0_H', 1.))

        ############################### (ADDED) hyperparameters for additional sparsity on W ###############################
        self.a0_W = float(kwargs.get('a0_W', 1.))
        self.b0_W = float(kwargs.get('b0_W', 1.))

        # hyperparameters for stochastic (natural) gradient
        self.t0 = float(kwargs.get('t0', 1.))
        self.kappa = float(kwargs.get('kappa', 0.5))

    def _init_components(self, n_feats):
        # variational parameters for components W
        self.nu_W = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_feats, self.n_components))
        self.rho_W = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_feats, self.n_components))

        # variational parameters for sparsity pi_H on H matrix
        self.alpha_pi_H = np.random.rand(self.n_components)
        self.beta_pi_H = np.random.rand(self.n_components)

        ############################### (ADDED) variational parameters for sparsity pi on W matrix ###############################
        self.alpha_pi_W = np.random.rand(self.n_components)
        self.beta_pi_W = np.random.rand(self.n_components)

    def _init_weights(self, n_samples):
        # variational parameters for activations H
        self.nu_H = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_samples))
        self.rho_H = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_samples))

    def fit(self, X):
        n_feats, n_samples = X.shape
        self._init_components(n_feats)
        self._init_weights(n_samples)
        self.good_k = np.arange(self.n_components)
        # randomly initalize binary mask (S_H)
        self.S_H = (np.random.rand(self.n_components, n_samples) > .5)
        ############################### (ADDED) initialize S_W too ###############################
        self.S_W = (np.random.rand(n_feats, self.n_components) > .5)

        self._ssmf_a(X)

        return self

    def _ssmf_a(self, X):
        self.log_ll = np.zeros((self.max_iter, self.burn_in+1))
        
        ############################ (ADDED)#############################
        self.good_k_list = []
        self.Epi_H_list = np.zeros((self.max_iter, self.n_components))
        self.Epi_W_list = np.zeros((self.max_iter, self.n_components))
        
        for i in range(self.max_iter):
            good_k = self.good_k
            if self.verbose:
                print('SSMF-A iteration %d\tgood K:%d:' % (i, good_k.size))
                sys.stdout.flush()
            eta = (self.t0 + i)**(-self.kappa)
            # initialize W, H, pi_H, pi_W
            W = np.random.gamma(self.nu_W[:, good_k],
                                1. / self.rho_W[:, good_k])
            H = np.random.gamma(self.nu_H[good_k], 1. / self.rho_H[good_k])
            pi_H = np.random.beta(
                self.alpha_pi_H[good_k], self.beta_pi_H[good_k])
            ############################### (ADDED) initialize pi_W too ###############################
            pi_W = np.random.beta(
                self.alpha_pi_W[good_k], self.beta_pi_W[good_k])

            for b in range(self.burn_in+1):
                # burn-in plus one actual sample
                self.gibbs_sample_S_H(X, W, H, pi_H, pi_W)
            ############################### (ADDED) gibbs sample S_W too ###############################
                self.gibbs_sample_S_W(X, W, H, pi_H, pi_W)

                self.log_ll[i, b] = _log_likelihood(
                    X, self.S_H[good_k], self.S_W[:, good_k], W, H, pi_H, pi_W)

                if self.verbose and b % 10 == 0:
                    sys.stdout.write('\r\tGibbs burn-in: %d' % b)
                    sys.stdout.flush()
            if self.verbose:
                sys.stdout.write('\n')
            self._update(eta, X, W, H)

           # Epi_H = self.alpha_pi_H[good_k] / (self.alpha_pi_H[good_k] +
            #                                   self.beta_pi_H[good_k])
            #Epi_W = self.alpha_pi_W[good_k] / (self.alpha_pi_W[good_k] +
             #                                  self.beta_pi_W[good_k])
                
            self.Epi_H_list[i] = self.alpha_pi_H / (self.alpha_pi_H +
                                               self.beta_pi_H)
            self.Epi_W_list[i] = self.alpha_pi_W / (self.alpha_pi_W +
                                               self.beta_pi_W)
            
            alive_pi_H = good_k[self.Epi_H_list[i][good_k] > self.Epi_H_list[i][good_k].max() * self.cutoff]
            alive_pi_W = good_k[self.Epi_W_list[i][good_k] > self.Epi_W_list[i][good_k].max() * self.cutoff]
            self.good_k = np.array(list(set(alive_pi_H) & set(alive_pi_W)))
            ########################## (ADDED)######################################
            self.good_k_list.append(self.good_k)


        pass

    def gibbs_sample_S_H(self, X, W, H, pi_H, pi_W, log_ll=None):
        good_k = self.good_k
        for i, k in enumerate(good_k):
            ############################### (modified) added S_W ###############################
            # X_neg_k: F x T
            X_neg_k = (W * self.S_W[:, good_k]).dot(H * self.S_H[good_k]) - np.outer(W[:, i] * self.S_W[:, k],
                                                                                     H[i] * self.S_H[k])
            ############################### (modified) added S_W ###############################
            # log_Ph: 1 x T
            log_Ph = np.log(pi_H[i] + EPS) + np.sum(X * np.log(X_neg_k +
                                                         np.outer(W[:, i] * self.S_W[:, k], H[i]) + EPS)
                                              - np.outer(W[:, i] * self.S_W[:, k], H[i]), axis=0)
            ############################### (modified) added S_W ###############################
            # log_Pt: 1 x T
            log_Pt = np.log(1 - pi_H[i] + EPS) + np.sum(X * np.log(X_neg_k + EPS),  # Added + EPS next to pi_H[i]
                                                        axis=0)
            # subtract maximum to avoid overflow
            max_P = np.maximum(log_Ph, log_Pt)
            ratio = np.exp(log_Ph - max_P) / (np.exp(log_Ph - max_P) +
                                              np.exp(log_Pt - max_P))
            self.S_H[k] = (np.random.rand(self.S_H.shape[1]) < ratio)
            if type(log_ll) is list:
                log_ll.append(_log_likelihood(
                    X, self.S_H[good_k], W, H, pi_H, pi_W))
        pass

    ############################### (added) new gibbs_sample_S_W function for gibbs sampling S_W ###############################
    def gibbs_sample_S_W(self, X, W, H, pi_H, pi_W, log_ll=None):
        good_k = self.good_k
        for i, k in enumerate(good_k):
            X_neg_k = (W * self.S_W[:, good_k]).dot(H * self.S_H[good_k]) - np.outer(W[:, i] * self.S_W[:, k],
                                                                                     H[i] * self.S_H[k])
            log_Ph = np.log(pi_W[i] + EPS) + np.sum(X * np.log(X_neg_k +
                                                               np.outer(W[:, i], H[i] * self.S_H[k]) + EPS)
                                                    - np.outer(W[:, i], H[i] * self.S_H[k]), axis=1)

            log_Pt = np.log(1 - pi_W[i] + EPS) + np.sum(X * np.log(X_neg_k + EPS),  # Added + EPS next to pi_W[i]
                                                        axis=1)
            # subtract maximum to avoid overflow
            max_P = np.maximum(log_Ph, log_Pt)
            ratio = np.exp(log_Ph - max_P) / (np.exp(log_Ph - max_P) +
                                              np.exp(log_Pt - max_P))
            self.S_W[:, k] = (np.random.rand(self.S_W.shape[0]) < ratio)


    def _update(self, eta, X, W, H):
        good_k = self.good_k
        X_hat = (W * self.S_W[:, good_k]).dot(H * self.S_H[good_k]) + EPS
        # update variational parameters for components W

    ############################### (modified) add self.S_W ###############################
        self.nu_W[:, good_k] = (1 - eta) * self.nu_W[:, good_k] + \
            eta * (self.a + (W * self.S_W[:, good_k]) *
                   (X / X_hat).dot((H * self.S_H[good_k]).T))
        self.rho_W[:, good_k] = (1 - eta) * self.rho_W[:, good_k] + \
            eta * (self.b + (H * self.S_H[good_k]
                             ).sum(axis=1) * self.S_W[:, good_k])

        # update variational parameters for activations H

    ############################### (modified) W.T -> (W * S_W).T  ###############################
        self.nu_H[good_k] = (1 - eta) * self.nu_H[good_k] + \
            eta * (self.c + H * self.S_H[good_k] *
                   (W * self.S_W[:, good_k]).T.dot(X / X_hat))
        self.rho_H[good_k] = (1 - eta) * self.rho_H[good_k] + \
            eta * (self.d + (W * self.S_W[:, good_k]).sum(axis=0)
                   [:, np.newaxis] * self.S_H[good_k])

        # update variational parameters for sparsity pi_H
        self.alpha_pi_H[good_k] = (1 - eta) * self.alpha_pi_H[good_k] + \
            eta * (self.a0_H / self.n_components +
                   self.S_H[good_k].sum(axis=1))
        self.beta_pi_H[good_k] = (1 - eta) * self.beta_pi_H[good_k] + \
            eta * (self.b0_H * (self.n_components - 1) / self.n_components
                   + self.S_H.shape[1] - self.S_H[good_k].sum(axis=1))

        ############################### (added) updates for pi_W ###############################
        self.alpha_pi_W[good_k] = (1 - eta) * self.alpha_pi_W[good_k] + \
            eta * (self.a0_W / self.n_components +
                   self.S_W[:, good_k].sum(axis=0))
        self.beta_pi_W[good_k] = (1 - eta) * self.beta_pi_W[good_k] + \
            eta * (self.b0_W * (self.n_components - 1) / self.n_components
                   + self.S_W.shape[0] - self.S_W[:, good_k].sum(axis=0))

    def transform(self, X):
        raise NotImplementedError('Wait for it')


def _log_likelihood(X, S_H, S_W, W, H, pi_H, pi_W):
    log_ll = scipy.stats.bernoulli.logpmf(S_H, pi_H[:, np.newaxis]).sum()
    log_ll += scipy.stats.bernoulli.logpmf(S_W, pi_W[np.newaxis, :]).sum()
    log_ll += scipy.stats.poisson.logpmf(round(X), ((W * S_W).dot(H * S_H) + EPS)).sum()

    return log_ll



################################### (ADDED) Helper functions below, all added ######################################

def get_post_means(obj):
    nu_W = obj.nu_W
    rho_W = obj.rho_W
    nu_H = obj.nu_H
    rho_H = obj.rho_H
    good_k = obj.good_k
    W_post_mean = np.zeros(nu_W[:, good_k].shape)
    for i in range(W_post_mean.shape[0]):
        for j in range(W_post_mean.shape[1]):
            W_post_mean[i,j] = nu_W[:, good_k][i][j] / rho_W[:, good_k][i][j]
            
    H_post_mean = np.zeros(rho_H[good_k].shape)
    for i in range(H_post_mean.shape[0]):
        for j in range(H_post_mean.shape[1]):
            H_post_mean[i,j] = nu_H[good_k][i][j] / rho_H[good_k][i][j]
    
    X_post_mean = (W_post_mean * obj.S_W[:, obj.good_k]).dot(H_post_mean * obj.S_H[obj.good_k])
    
    return W_post_mean, H_post_mean, X_post_mean

import matplotlib.pyplot as plt

def draw_two_matrices(X1, X2, shrink1=1, shrink2=0.3, first='X1', second='X2'):
    
    fig, axes = plt.subplots(ncols=2, figsize=(15,5))
    ax1, ax2 = axes
    im1, im2 = ax1.imshow(X1), ax2.imshow(X2)
    plt.colorbar(im1, ax=ax1, shrink=shrink1)
    plt.colorbar(im2, ax=ax2, shrink=shrink2)
    ax1.title.set_text(first)
    ax2.title.set_text(second)
    plt.show()
    return

def draw_S_H_S_W(obj):
    fig, axes = plt.subplots(ncols=3, figsize=(15,5))
    ax1, ax2, ax3 = axes
    im1, im2, im3 = ax1.imshow(obj.S_W[:, obj.good_k], aspect="auto", interpolation='none'), ax2.imshow(obj.S_H[obj.good_k], aspect="auto", interpolation='none')\
    , ax3.imshow(obj.S_W.dot(obj.S_H), aspect="auto", interpolation="none")
    ax1.title.set_text('S^W ({} x {})'.format(obj.S_W[:, obj.good_k].shape[0], obj.S_W[:, obj.good_k].shape[1]))
    ax2.title.set_text('S^H ({} x {})'.format(obj.S_H[obj.good_k].shape[0], obj.S_H[obj.good_k].shape[1]))
    ax3.title.set_text('S^W . S^H ({} x {})'.format(obj.S_W.dot(obj.S_H).shape[0], obj.S_W.dot(obj.S_H).shape[1]))
    plt.colorbar(im3, ax=ax3)

    plt.show()
    
    return

def plot_pi(obj):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    labels = np.arange(0, obj.n_components,1)
    for i in range(obj.n_components):
            axes[0].plot(obj.Epi_W_list[:,i], label="pi_" + str(labels[i]))
    axes[0].title.set_text("Value of Pi_W's across iterations")
    axes[0].legend()

    for i in range(obj.n_components):
        axes[1].plot(obj.Epi_H_list[:,i], label="pi_" + str(labels[i]))
    axes[1].title.set_text("Value of Pi_H's across iterations")
    axes[1].legend()
    return


    
def rrmse(X, X_hat):
    return np.sqrt( np.sum((X_hat-X)**2).sum() / np.sum(X**2).sum() )

def evaluate(X, X_hat):
    mse = np.mean((X-X_hat)**2).mean()
    rrmse_val = rrmse(X, X_hat)
    print("mse: {} \t rrmse:{}".format(mse, rrmse_val))
    return mse, rrmse
