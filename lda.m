function [stat_values,cls_test,prob]= lda( train,test,labels,cls_ver )

[cls_test,err,prob]=classify(test,train,labels,'linear');
    
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

stat_values=[specificity,sensitivity,VN_lda,FP_lda,VP_lda,FN_lda,accuracy];
end

