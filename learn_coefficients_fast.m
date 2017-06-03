function Sout = learn_coefficients_fast(B, X, alpha, gamma, Linit, Sinit)

% Peng peixi
%Unsupervised Cross-Dataset Transfer Learning for Person Re-identification
%cvpr 2016

warning('off', 'MATLAB:divideByZero');
L=Linit;

use_Sinit= false;
if exist('Sinit', 'var')
    use_Sinit= true;
end

Sout= zeros(size(B,2), size(X,2));
BtB = B'*B;
BtX = B'*X;



for i=1:size(X,2)
    
    
    if use_Sinit
         Temp=zeros(size(Sout,1),1);
         
         indext=find(L(:,i)~=0);
      for j=1:length(indext)
     if indext(j)<i
         
        Temp=Temp+alpha*L(indext(j),i).*Sout(:,indext(j));
     elseif indext(j)>i
        Temp=Temp+alpha*L(indext(j),i).*Sinit(:,indext(j));
     end
     end
   
 Sout(:,i)=(BtB+alpha*L(i,i)*eye(size(B,2))+gamma*eye(size(B,2)))\(BtX(:,i)-Temp);
 

        
%         [Sout(:,i), fobj]= ls_featuresign_sub (B, S, X(:,i), BtB, BtX(:,i), L, i, alpha, gamma, sinit);
    else
        Temp=zeros(size(Sout,1),1);
        indext=find(L(:,i)~=0);
        for j=1:length(indext)
             if indext(j)~=i
                Temp=Temp+alpha*L(indext(j),i).*Sout(:,indext(j));
             end
        end
        Sout(:,i)=(BtB+alpha*L(i,i)*eye(size(B,2))+gamma*eye(size(B,2)))\(BtX(:,i)-Temp);
    end
end
warning('on', 'MATLAB:divideByZero');
Sout(isnan(Sout))=0;
return;
