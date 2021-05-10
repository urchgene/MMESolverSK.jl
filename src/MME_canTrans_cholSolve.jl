##########################################################################################
###### This solves MT MME using canonical transformation to convert it problen to 
###### uncorrelated univariate systems thus solving the computational burden associated
###### with large MME BLUP. 
###### Developer: Uche Godfrey Okeke
###### See Ducrocq and Chapuis (1997) and K. Meyer (1991, 1985) for details
###### This is for equal design matrices (X,Z) problem ie same traits, same indvs...
##########################################################################################


using LinearAlgebra, DataFrames
using PositiveFactorizations, SparseArrays


function solveChol_MT(X, Z, Vg, Ve, n, d, Y, Ainv, linenames)

	#### Get Q-matrix for transformation from Ducrocq and Besbes (1993)
	L = cholesky(Ve).L
	D = eigvals(L'*inv(Vg)*L)
	U = eigvecs(L'*inv(Vg)*L)
	Q = U'*inv(L)
	
	### transform data
	d = Int64.(d)
	In = one(rand(n, n)); y = vec(Y); yQ = kron(Q, In) * y; YQ = reshape(yQ, n, d)

	#### define matrices to hold gebvs and betas...
	uhat = zeros(size(Z,2), d); beta = zeros(size(X,2), d);


	#### Split systems of equations into univariate transformed systems
	
	for i in 1:d	
	Sigma = D[i]*Ainv;
	XtX = X'*X; XtZ = X'*Z; ZtX = Z'*X; ZtZG = Z'*Z + Sigma;
	C = hcat(vcat(XtX, ZtX), vcat(XtZ, ZtZG));
	RHS = vcat(X'*YQ[:,i], Z'*YQ[:,i])

        ## Clean memory first...
        XtX = 0; XtZ = 0; ZtX = 0; ZtZG = 0; Rinvs = 0; Sigma = 0; In = 0; GC.gc()

	#### Solve by Cholesky...
        F, l = ldlt(Positive, C); GC.gc()
        theta = F\RHS;
	beta[:,i] = theta[1:size(X,2)];
        uhat[:,i] = theta[size(beta,1)+1: end];        
        end
        
	gg1 = size(beta,1); gg2 = size(uhat, 1);

	### transform solutions back to original scale
	uhat = vec(uhat); beta = vec(beta); Inu = sparse(1.0I, gg2, gg2); Inb = sparse(1.0I, gg1, gg1);
	uhat = kron(inv(Q), Inu) * uhat; beta = kron(inv(Q), Inb) * beta;
	uhat = reshape(uhat, :, d); beta = reshape(beta, :, d);
	uhat = DataFrame(Lines=linenames, Uhat=uhat);

	m11 =  Dict(
                :uhat => uhat,
                :beta => beta)

	return(m11)

end
