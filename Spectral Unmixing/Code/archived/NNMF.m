function [Ac, H] = NNMF(Y, A0, H0, epsilon, max_iter)

Ac = A0;
shape_A = size(Ac);
Y0 = Y;

tol = 0.01;%tolerance for convergance
alpha_1 = 0.01;
alpha_2 = 0.1;

A = A0;
H = H0;

for i = 1:max_iter 
    disp(i)
    if mod(i, 20) == 0
        if i == 20
            rcond((transpose(A) *(A+epsilon)))
            Ac = A;
            A = A0;
        else
            A
            rcond((transpose(A) *(A+epsilon)))
            Ac = Ac*A;
            A = A0;
        end
        Y = H; %This could be an issue for autoflourescence where Y and H don't have the same dims initially
        Ac
        %if rcond((transpose(Ac) *(Ac+epsilon)))<0.001
        %   disp('Singular Matrix')
        %   Ac = last_good_Ac
        %   break 
        %else
        %   last_good_Ac = Ac 
        %end
        H = inv(transpose(Ac) * (Ac+epsilon))*transpose(Ac)*Y0;
        H(H<=0) = epsilon;
    end
    H_int = (transpose(A)*Y-alpha_1);
    H_int(H_int<=0) = epsilon;

    H = H.* H_int ./ ((transpose(A)*A)*H+epsilon);
    A_int = (H*transpose(Y)-alpha_2);
    A_int(A_int<=0) = epsilon;
    A = A .* transpose(A_int ./ (transpose(A*(H*transpose(H)))+epsilon));
    A = A./ sum(A);
    %inv(Ac*A)
    if i > 20
        if norm(A - eye(shape_A(2)))<tol
            disp('Converged')
            Ac
            break
        end
    end
end

H = inv(transpose(Ac) * (Ac+epsilon))*transpose(Ac)*Y0;
H(H<0) = 0;