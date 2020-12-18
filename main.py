import streamlit as st

import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import deepdish as dd

def lhs_sample(N=10, D=2):
        '''Sample using hypercube'''
        #I will provide samples in the unit hypercube
        bins = np.linspace(0, 1, N, endpoint=False)
        interval = 1/N 
        centroids = bins + interval/2
        lhs_centroids = np.column_stack(tuple(np.random.permutation(centroids) for i in range(D)))
        sample = lhs_centroids + np.random.uniform(low=-interval/2, high=interval/2, size=(N, D))
        return bins, lhs_centroids, sample[np.argsort(sample.T[0,:])]
class utlhs():                    
    def __init__(self, N=3, D=2):
        self.N = N
        self.D = D
        self.bins, self.lhs_centroids, self.samples = self.lhs_sample(N=self.N, D=self.D)
        return None
    
    def lhs_sample(self, N=10, D=2):
        '''Sample using hypercube'''
        #I will provide samples in the unit hypercube
        bins = np.linspace(0, 1, N, endpoint=False)
        interval = 1/N 
        centroids = bins + interval/2
        lhs_centroids = np.column_stack(tuple(np.random.permutation(centroids) for i in range(D)))
        sample = lhs_centroids + np.random.uniform(low=-interval/2, high=interval/2, size=(N, D))
        return bins, lhs_centroids, sample[np.argsort(sample.T[0,:])]

    def upsample(self, factor=2, update=True):
        if factor < 2:
            raise AssertionError("The upsampling needs at least doubling the samples (i.e. factor=2)")

        # Prepare new grid and mask
        newN = int(factor*self.N)
        newbins = np.linspace(0, 1, newN, endpoint=False)
        interval = 1/newN

        # Create a D-dimensional mask to flag bins already populated
        mask = np.ones(tuple(newN for i in range(self.D)), dtype=bool)
        # Find the centroids of samples already available
        newlhs_centroids = newbins + interval/2
        
        # Find the centroids of samples already available
        # D-uple of 1darray (length N), with bin coordinate of lhs samples 
        prev_idxs = tuple(np.searchsorted(newbins, self.samples[:,i])-1 for i in range(self.D))
        # Fill the mask with False for slices taken
        mask[prev_idxs] = False
        
        # This is tricky:
        # Construct new edge-oid (lower edge on each dimension) of sampled cells by making random permutation of bins not already taken.
        new_edgeoids = np.column_stack(tuple(np.random.permutation(newbins[np.all(mask, axis=tuple(j for j in range(self.D) if j is not i))]) for i in range(self.D)))
        # then add a random displacement 
        
        new_samples = new_edgeoids + np.random.uniform(low=0, high=interval, size=(int(factor-1)*self.N, self.D))

        if update:
            # Update class properties with new values
            self.bins = newbins
            new_centroids = new_edgeoids + interval/2 * np.ones(shape=(int(factor-1)*self.N, self.D))
            self.N = newN
            self.lhs_centroids = np.concatenate((self.lhs_centroids, new_centroids), axis=0)
            self.samples = np.concatenate((self.samples, new_samples), axis=0)
            return
    
        else:
            new_centroids = new_edgeoids + interval/2 * np.ones(shape=(int(factor-1)*self.N, self.D))
            lhs_centroids = np.concatenate((self.lhs_centroids, new_centroids), axis=0)
            samples = np.concatenate((self.samples, new_samples), axis=0)
            return newN, self.D, newbins, lhs_centroids, samples
    
    def downsample(self, factor):
        if not self.N%factor==0:
            raise AssertionError("You want an integer fraction of the original samples")
        raise NotImplementedError('''I believe no algorithm exist to "coarse-grain". Get in touch if you find one''')
            
    def higher_D(self, dims=np.array([-1]), dtype=int):
        '''Extend each sample by adding i-th coordinates for i in dims'''
        dims = dims%(self.D+1) # Clean -1's as some function cannot handle it
        Dcoord = np.shape(dims)[0]
        self.D+=Dcoord
        bins = np.linspace(0, 1, self.N, endpoint=False)
        interval = 1/self.N 
        centroids = bins + interval/2
        # New centroids coordinates
        centroids_newcoord = np.column_stack(tuple(np.random.permutation(centroids) for i in range(Dcoord)))
        # New sample coordinates
        samples_newcoord = centroids_newcoord + np.random.uniform(low=-interval/2, 
                                                                  high=interval/2,
                                                                  size=(self.N,Dcoord))
        
        self.lhs_centroids = np.insert(self.lhs_centroids, obj=dims, values=centroids_newcoord, axis=1)
        self.samples = np.insert(self.samples, obj=dims, values=samples_newcoord, axis=1)
        return
    
    def lower_D(self, dims=np.array([-1], dtype=int)):
        '''Reduce the dimensions of the lhs, by erasing coordinates tupled in dims (default only last coordinate).'''
        dims = dims%self.D # Clean -1's as some function cannot handle it
        Dcoord = np.shape(dims)[0]
        self.D -= Dcoord
        bins = np.linspace(0, 1, self.N, endpoint=False)
        interval=1/self.N
        centroids = bins + interval/2
        self.lhs_centroids = np.delete(self.lhs_centroids, obj=dims, axis=1)
        self.samples = np.delete(self.samples, obj=dims, axis=1)
        return

    def extend_domain(self, ncells=1):
        '''Extend the domain, keeping the previous samples. Effectively produces 
        a blocky lhs: lowest block is composed of previous samples. Highest is made of new samples, 
        effectively a new translated lhs of size ncells. The result is rescaled to the unit hypercube for consistency.
        The user can rescale it to his/her own needs'''
        # Preparing new samples
        extD = self.D
        extN = ncells
        #Create samples, discard bins
        __, extlhs_centroids, extsamples = self.lhs_sample(N=ncells, D=self.D)
        
        # Do the same to centroids and samples
        for oldpoints, newpoints in zip([self.lhs_centroids, self.samples],[extlhs_centroids, extsamples]):
            # Scale down 
            newpoints*=extN/(extN+self.N)
            # and shift
            newpoints+=self.N/(extN+self.N)
            # To fit in space made from old ones
            oldpoints*=self.N/(extN+self.N)
        
        # Update class status
        self.N += extN
        self.bins = np.linspace(0, 1, self.N, endpoint=False)
        self.lhs_centroids = np.concatenate((self.lhs_centroids, extlhs_centroids), axis=0)
        self.samples = np.concatenate((self.samples, extsamples), axis=0)
        return
    
    def restrict_domain(self):
        raise NotImplementedError("Yet to be implemented")

    def plot_sample(self, axes=None, plotD=2, grid=True):
        '''Plot the projection of the hypercube samples onto the i-th and j-th coordinate'''
        assert plotD<=self.D
        if plotD==1:
            if axes==None: axes=(0,)
            fig = plt.figure(1, (5,1))
            ax = fig.gca()
            ar = np.arange(self.N) # just as an example array
            ax.scatter(self.samples[:,axes[0]], np.zeros((self.N,)), c='k')
            ax.set_xlabel(f"$x_0$")
            ax.set_xlim(0,1)
            ax.tick_params(axis='y',
                           which='both',    
                           left=False,      
                           right=False,         
                           labelright=False,
                           labelleft=False) 
            if grid:
                for b in self.bins:
                    ax.axvline(b, c='k')
        elif plotD==2:
            if axes==None: axes=(0,1)
            fig = plt.figure(1, (5,5))
            ax = fig.gca()
            ax.scatter(self.samples[:,axes[0]], self.samples[:,axes[1]], c='k')
            ax.set_xlabel(f"$x_{axes[0]}$")
            ax.set_ylabel(f"$x_{axes[1]}$")
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            if grid:
                for b in self.bins:
                    ax.axvline(b, c='k')
                    ax.axhline(b, c='k')
        elif plotD==3:
            if axes==None: axes=(0,1,2)
            fig = plt.figure(1, (5,5))
            ax = Axes3D(fig) 
            ax.scatter(self.samples[:,axes[0]], self.samples[:,axes[1]],self.samples[:,axes[2]], c='k')
            ax.set_xlabel(f"$x_{axes[0]}$")
            ax.set_ylabel(f"$x_{axes[1]}$")
            ax.set_zlabel(f"$x_{axes[2]}$")
            ax.set_xlim(-0.1,1.1)
            ax.set_ylim(-0.1,1.1)
            ax.set_zlim(-0.1,1.1)
        return fig,ax
    
    def save(self, path):
        '''Minimal saving. This overrides the current instance status'''
        dd.io.save(path, dict(samples=self.samples, 
                              lhs_centroids=self.lhs_centroids,
                              N=self.N, D=self.D,
                              bins= np.linspace(0, 1, self.N, endpoint=False)))
        return path
    
    def load(self, path):
        '''Minimal loading. This overrides the current instance status'''
        file = dd.io.load(path)
        for k in ['N','D','bins','lhs_centroids','samples']:
            setattr(self, k, file[k])
        return 

total_points = st.slider("Number of cells in Hypercube 1D", 1, 100, 2)

with st.echo(code_location='below'):  

    D=1

    # Start with a 1D sample
    sampler = utlhs(N=total_points, D=D)

    fig, ax = sampler.plot_sample(plotD=1, grid=True)
    st.pyplot(fig)

with st.echo(code_location='below'):  

    D=2

    # Start with a 1D sample
    sampler = utlhs(N=total_points, D=D)
        
    fig, ax = sampler.plot_sample(plotD=2, grid=True)
    st.pyplot(fig)

with st.echo(code_location='below'):  

    D=3

    # Start with a 1D sample
    sampler = utlhs(N=total_points, D=D)

    fig, ax = sampler.plot_sample(plotD=3, grid=True)
    st.pyplot(fig)

        
