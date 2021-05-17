##########################################################################################
###### This solves MT MME using canonical transformation to convert it problen to 
###### uncorrelated univariate systems thus solving the computational burden associated
###### with large MME BLUP. 
###### Developer: Uche Godfrey Okeke
###### See Ducrocq and Chapuis (1997) and K. Meyer (1991, 1985) for details
###### This is for equal design matrices (X,Z) problem ie same traits, same indvs...
##########################################################################################


using LinearAlgebra
using PositiveFactorizations, DataFrames
using SparseArrays, IterativeSolvers

function solvePCG_MT(X, Z, Vg, Ve, n, d, Y, Ainv, linenames)

	#### Get Q-matrix for transformation from Ducrocq and Besbes (1993)
        Ve = Ve + 0.0001I; Vg = Vg + 0.0001I;
	#L = cholesky(Ve).L
        L = cholesky(Positive, Ve).L
	D = eigvals(L'*inv(Vg)*L)
	U = eigvecs(L'*inv(Vg)*L)
	Q = U'*inv(L)
	
	### transform data
	d = Int64.(d)
	In = sparse(1.0I, n, n); 
	y = vec(Y); yQ = kron(Q, In) * y; YQ = reshape(yQ, n, d)

	#### define matrices to hold gebvs and betas...
	uhat = zeros(size(Z,2), d); beta = zeros(size(X,2), d);


	#### Split systems of equations into univariate transformed systems
	
	for i in 1:d	
	Sigma = D[i]*Ainv;
	XtX = X'*X; XtZ = X'*Z; ZtX = Z'*X; ZtZG = Z'*Z + Sigma;
	C = hcat(vcat(XtX, ZtX), vcat(XtZ, ZtZG)); C = sparse(C);
	RHS = vcat(X'*YQ[:,i], Z'*YQ[:,i])

        ## Clean memory first...
        XtX = 0; XtZ = 0; ZtX = 0; ZtZG = 0; Rinvs = 0; Sigma = 0; In = 0; GC.gc()

	#### Solve by PCG iteration on bicgstabl...
        #P = ilu(C, τ = 0.1); 
	GC.gc()
        #theta = bicgstabl(C, RHS, 2, Pl = P, max_mv_products = 2000);
	theta = bicgstabl(C, RHS, 2, Pl = Diagonal(C), max_mv_products = 2000);
	beta[:,i] = theta[1:size(X,2)];
        uhat[:,i] = theta[size(beta,1)+1: end];        
        end
        
        gg1 = size(beta,1); gg2 = size(uhat, 1);

	### transform solutions back to original scale, In is just 1...change the sizes as appropriate to match dim
	uhat = vec(uhat); beta = vec(beta); Inu = sparse(1.0I, gg2, gg2); Inb = sparse(1.0I, gg1, gg1);
	uhat = kron(inv(Q), Inu) * uhat; beta = kron(inv(Q), Inb) * beta;
	uhat = reshape(uhat, :, d); beta = reshape(beta, :, d);
	uhat = DataFrame(uhat, :auto); df = DataFrame(Lines=linenames);
        EBVs = hcat(df, uhat);

	m11 =  Dict(
                :uhat => EBVs,
                :beta => beta)

	return(m11)

end
