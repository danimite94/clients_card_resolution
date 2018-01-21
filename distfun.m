function [stat_values,cls_test]= distfun(opt,train,test,labels,cls_ver )

%Matriz treino e teste tem observaçoes nas linhas e variaveis em cada coluna
%funçoes para realizaçao do clustering

VP_lda=0;
FP_lda=0;
FN_lda=0;
VN_lda=0;
pos_mean=[];
neg_mean=[];
if (opt==1) %euclidean distance
    pos_ind=find(labels==1);
    neg_ind=find(labels==0);
    
    for i=1:size(train,2)
        pos_mean=[pos_mean mean(train(pos_ind,i))];
        neg_mean=[neg_mean mean(train(neg_ind,i))];
    end
    
    
    for k=1:size(test,1)
        distance1=sqrt(sum((pos_mean-test(k,:)).^2));
        
        distance2=sqrt(sum((neg_mean-test(k,:)).^2));
        
        if (distance1<=distance2)
            cls_test(k,1)=1;
            
        else
            cls_test(k,1)=0;
        end
    end
    
elseif(opt==2) %mahalanobis
    
    [cls_test,err,posterior]=classify(test,train,labels,'mahalanobis');
end  
    for i=1:length(cls_test)
        if( (isequal(cls_ver(i),1)==0) && (isequal(cls_test(i),1)==1))
            FP_lda=FP_lda+1;
        elseif((isequal(cls_ver(i),1)==1) && (isequal(cls_test(i),1)==0))
            FN_lda=FN_lda+1;
        elseif((isequal(cls_ver(i),1)==1) && (isequal(cls_test(i),1)==1))
            VP_lda=VP_lda+1;
        elseif((isequal(cls_ver(i),1)==0) && (isequal(cls_test(i),1)==0))
            VN_lda=VN_lda+1;
        end
        
    end

specificity=VN_lda/(VN_lda+FP_lda);
sensitivity=VP_lda/(VP_lda+FN_lda);
accuracy=(VP_lda+VN_lda)/(VP_lda+FP_lda+FN_lda+VN_lda);
% FP_rate=FP_lda/(VN_lda+FP_lda);

stat_values=[specificity,sensitivity,VN_lda,FP_lda,VP_lda,FN_lda,accuracy];

end
%fazer mahalanobis
