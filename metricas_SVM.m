%Modelos resultantes do SVM
load('svmlin.mat'); %SVMModel_lin presente no documento projectcode
load('svmrbf.mat')  %SVMModel_rbf presente no documento projectcode


[impdata,var] = xlsread('clientsdataset.xls');
varname = {}; varname = [varname var{2,2:(end-1)}];
data.X = impdata(:,2:end-1)';
data.y = impdata(:,end)';
data.num_data = size(data.X,2);
data.dim = size(data.X,1);
data.name = 'default_credit_cards_taiwan';
clear impdata, clear var

%preprocess
data = stdscale(data);

p = zeros(size(data.X,1),1);
feat_2keep_index = [];
chi_sq = [];
for i=1:size(data.X,1)
    [p(i),ANOVATAB,~]=kruskalwallis(data.X(i,:),data.y);
    if p(i)<0.01
        feat_2keep_index = [feat_2keep_index i];
        chi_sq = [chi_sq ANOVATAB{18}];
    end
    close all
end
[chi_sq,n_order]=sort(chi_sq,'descend');
feat_2keep_index=feat_2keep_index(n_order);
data.X = data.X(feat_2keep_index,:);
varname = varname(feat_2keep_index);
R=corrcoef(data.X');
feat_2keep_index = 1:length(varname);
for i=(round(length(varname)/2)+1):length(varname)
    isremoved = false;
    for j=1:round(length(varname)/2)
        if R(i,j) > 0.10
            isremoved = true;
        end
    end
    if isremoved == true
        a=find(feat_2keep_index==i);
        feat_2keep_index = feat_2keep_index([1:(a-1) (a+1):end]);
    end
end

data.X = data.X(feat_2keep_index,:);
varname = varname(feat_2keep_index);

[model_3,score_3,latent_3,~,explained_3] = pca(data.X', 'NumComponents',3);

% SVM
X = score_3;
y = data.y';
clearvars -except X y LIN_MODELS RBF_MODELS

for i=1:length(LIN_MODELS)
    newy=predict(LIN_MODELS{i},X);
    C=confusionmat(y,newy);
    accuracy = C(1,1)+C(2,2)/(C(1,1)+C(2,2)+C(1,2)+C(2,1))
    sv = LIN_MODELS{i}.SupportVectors;
    ind_0 = find(y==0);
    ind_1 = find(y==1);
    figure
    plot3(X(ind_0,1),X(ind_0,2),X(ind_0,3),'ro')
    hold on
     plot3(X(ind_1,1),X(ind_1,2),X(ind_1,3),'go')
    plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
    legend('0','1','Support Vector')
    hold off
    pause()
end

for i=1:length(RBF_MODELS)
    newy=predict(LIN_MODELS{i},X);
    C=confusionmat(y,newy);
    accuracy = C(1,1)+C(2,2)/(C(1,1)+C(2,2)+C(1,2)+C(2,1))
    sv = LIN_MODELS{i}.SupportVectors;
    ind_0 = find(y==0);
    ind_1 = find(y==1);
    figure
    plot3(X(ind_0,1),X(ind_0,2),X(ind_0,3),'ro')
    hold on
     plot3(X(ind_1,1),X(ind_1,2),X(ind_1,3),'go')
    plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
    legend('0','1','Support Vector')
    hold off
    pause()
end
