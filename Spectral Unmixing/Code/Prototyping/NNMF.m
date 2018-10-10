function [Ac, H] = NNMF(Y, A0, epsilon)

Ac = A0;
H0 = Y; 
Y0 = Y;

tol = 0.01;%tolerance for convergance
alpha_1 = 0.01;
alpha_2 = 0.1;

A = A0;
H = H0;

%opt = statset('MaxIter', 200);
%[A, H] = nnmf(Y, num_channels, 'algorithm', 'als', 'w0', A0, 'h0', H0, 'options', opt); %'mult' is inferior to 'als'?
for i = 1:200 
    if mod(i, 20) == 0
        if i == 20
            Ac = A;
            A = A0;
        else
            Ac = Ac*A;
            A = A0;
        end
        Y = H;
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
    %disp(norm(A - eye(3)))
    if norm(A - eye(3))<tol
        disp('Converged')
        break
    end
end

%Ac = Ac./sum(Ac, 2)
H = inv(transpose(Ac) * (Ac+epsilon))*transpose(Ac)*Y0;