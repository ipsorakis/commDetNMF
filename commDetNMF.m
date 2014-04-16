function [P,g,W,H] = commDetNMF(V,max_rank,W0,H0)
% Implements the community detection methodology presented in 
% "Overlapping Community Detection using Nonnegative Matrix Factorization", Physical Review E 83, 066114 (2011)
% by Ioannis Psorakis, Stephen Roberts, Mark Ebden and Ben Sheldon.
% University of Oxford, United Kingdom.
%
% Inputs :
% V: NxN adjacency matrix
% max_rank: a prior estimation on the maximum number of communities
% (optional).
% W0,H0: prior values for W,H factors (optional)
% Outputs :
% 
% P: NxK matrix where each element Pik represents the (normalised)
% soft membership score of node-i to each community-k. P captures the OVERLAPPING
% community structure.
%
% g: groups cell array. Each element g{k} contains the node indices of
% community k. Each node i is placed in the community g{k} for which it has
% the maximum Pnk. Therefore g describes the NON-OVERLAPPING community
% structure.
%
% W,H are the NMF factors (normalised from 0 to 1)
%
% NOTES:
% ------
% - In many cases the algorithm gives better results if the diagonals of the adjacency
% matrix are non-zero, for example Aii = degree(i)
% - having an estimation on the upper bound of community numbers "max_rank"
% will significantly increase the performance of the algorithm.
% - P is based on W with zero columns removed.
% - Due to the coordinate descend optimisation scheme (see "Automatic Relevance Determination in Nonnegative Matrix
% Factorization", SPARS '09 by Tan and Fevotte), the NMF result is
% initialisation depend. You may have to run the algorithm multiple times if you
% didn't get the desired result from the first time.
% - email: ioannis.psorakis@eng.ox.ac.uk for questions/bug reports/comments!
%============

if (nargin < 2)
    max_rank = ceil(size(V,1)/2);
end;

N = size(V,1);

%remove non-connected nodes
active_nodes = or(sum(V)~=0,transpose(sum(V,2)~=0));
active_node_indices = find(active_nodes);
non_active_nodes = ~active_nodes;
non_active_node_indices = find(non_active_nodes);

if sum(non_active_nodes)>0
   V = V(active_node_indices,active_node_indices);
   fprintf('\nNOTE: the following non-connected nodes are removed from the graphs:\n')
   disp(non_active_node_indices)
end

%now scale the incoming matrix to lie in [0,1] for all elements
X = scale01(V);

%set up the initial W,H matrices
if (nargin < 3)
    W0 = rand(size(X,1),max_rank);
end;
if (nargin < 4)
    H0 = rand(max_rank,size(X,2));
end;

%now run the nmf with shrinkage priors
[W, H] = nmfBayesMap(X, 1000,W0,H0);

%allocate using greedy group selection
[a,b] = max(W');

%scale the allocations matrix
P = W ./ repmat(sum(W,2),1,max_rank);
%P = H' ./ repmat(sum(W,2),1,max_rank);

P(:,sum(P)==0)=[];

%now go through the groups finding unique sets
if max_rank>1
m=1;
for n=1:max_rank,
    f=find(b==n);
    if (~isempty(f))
        g{m}=f;
        m=m+1;
    end
end
g = g';

else
    g = (1:size(V,1))';
end

% recover original indices
K = length(g);
number_of_disconnected_nodes = sum(non_active_nodes);
if number_of_disconnected_nodes>0
        
    Pnew = zeros(N,size(P,2));
        
    % fix active
    for k=1:K
       g{k} = active_node_indices(g{k});
    end
    Pnew(active_nodes,:) = P;
    
    % fix inactive
    for k=K+1:K+number_of_disconnected_nodes
       g{k} = non_active_node_indices(k-K);
       Pnew(g{k},:) = 0;
    end    
    
    P = Pnew;
end

end
%--------------
function [W, H, outStruct] = nmfBayesMap(V,n_iter,W,H)
%[W, H, outStruct] = nmfBayesMap(V,n_iter,W,H,b0)
%performs MAP Bayesian inference for W,H such that V ~ W*H
%inputs : V - data, n_iter - number of iterations, W,H - initial matrices
%outputs : outStruct with fields .like [data likelihood]
%                                .evid [posteriorevidence]
%                                .invbeta [1/beta_k] shrinkage parameters


%basic check on number args in
if ~exist('n_iter','var')
  n_iter = 100;
end

%verbose variable to print reassuring dots every PRD iterations...
PRD = round(n_iter/5);

%set up various variables
[F,N] = size(V);
K = size(W,2);
a0 = 5; b0 = 2; %vanilla hyperpriors : these can be altered as required
a = a0*ones(K,1);
b = b0*ones(K,1);

%Pre-allocate output struct variables
outStruct.like = zeros(1,n_iter);
outStruct.evid = zeros(1,n_iter);
outStruct.invbeta = zeros(K,n_iter);

%Initialise the hyperprior
invbeta = ones(K,1);
V_ap = W*H;
a_post = N/2 + F/2 + a - 1;
b_post = sum(H.^2,2)/2 + sum(W.^2,1)'/2 + b;

%Initial likelihood, posterior evidence etc
like = sum(sum(V.*log(V./V_ap)-V+V_ap)); %data likelihood
outStruct.like(1) = like;
outStruct.evid(1) = like + sum(b_post./invbeta) + sum(a_post.*log(invbeta)); %posterior evidence
outStruct.invbeta(:,1)  = invbeta; %1/beta_k

%avoid numerical overflows - sqrt to make consistent with nmf code of Tan & Fevotte
epsilon = sqrt(eps);

%pre-define for speedup
I = ones(F,N);

for (iter = 2:n_iter) %main loop
    H = H ./ (W' * I + H./repmat(invbeta ,1,N) + epsilon  ) .* (W' *  (V ./ (W*H))); %update H
    b_post = sum(H.^2,2)/2 + sum(W.^2,1)'/2 + b; %update posteriot hyperprior
    invbeta = b_post./a_post;
    W = W ./ (I * H' + W./repmat(invbeta',F,1) + epsilon) .* ( (V./(W*H)) * H'); %update W
    V_ap = W*H; %estimate of V
    like = sum(sum(V.*log(V./V_ap)-V+V_ap)); %get data likelihood
    outStruct.like(iter) = like;
    outStruct.evid(iter) = like + sum(b_post./invbeta) + sum(a_post.*log(invbeta)); %get posterior evidence
    outStruct.invbeta(:,iter)  = invbeta; %store 1/beta_k
    if (mod(iter,PRD) == 0) %print reassuring dots...
        fprintf('.');
    end
end
fprintf('\n');

end

%--------------
function y = scale01(x)
  
  y = x - min(min(x)) + eps;
  y = y/max(max(y));
  
return
end