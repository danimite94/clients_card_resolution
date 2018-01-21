function [stat_values,cls_test]=k_nn_loo(test,k,cls_ver)

cls_test=zeros(size(test,1),1);

for test_r=1:size(test,1)
    dis=zeros(size(test,1),2);
    tot_min=zeros(k,2);
    for train_r=1:size(test,1)
        if (train_r==test_r)
            continue;
        else
            dis(train_r,:)=[sqrt(sum((test(test_r,:)-test(train_r,:)).^2)) cls_ver(train_r)];
            
        end
    end
    
    sort_asc = sortrows(dis,1); %pre defined as 'ascend'
    for n=1:k
        tot_min(n,:)=[sort_asc(n+1,1) sort_asc(n+1,2)];
    end
    if sum(tot_min(:,2))>k/2
        cls_test(test_r,1)=1;
    end
end

VP_lda=0;
FP_lda=0;
FN_lda=0;
VN_lda=0;

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
% FP_rate=FP_lda/(TN_lda+FP_lda);

stat_values=[specificity,sensitivity,accuracy,VN_lda,FP_lda,VP_lda,FN_lda];

end
   


