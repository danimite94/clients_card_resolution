[impdata,var] = xlsread('clientsdataset.xls');
varname = {}; varname = [varname var{2,2:(end-1)}];
data.X = impdata(:,2:end-1)';
data.y = impdata(:,end)';
data.num_data = size(data.X,2);
data.dim = size(data.X,1);
data.name = 'default_credit_cards_taiwan';
clear impdata, clear var

% normalize all the dataset
data = stdscale(data);

% Pattern Recognition Phases:
% |> Preprocessing
% isolate the population and extract it from the background
% Graphical Model Inspection (boxplot)




% Kruskal-Wallis Test
% Used for Feature Ranking. Sorts the feature values and
% assigns ordinal ranks.
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

% Set new order according to chi_sq (highest to lowest)
[chi_sq,n_order]=sort(chi_sq,'descend');
feat_2keep_index=feat_2keep_index(n_order);
data.X = data.X(feat_2keep_index,:);
varname = varname(feat_2keep_index);

f1 = figure('Name','Variables ordered accordingly to \chi^2 ');
h=uitable(f1,'Data',chi_sq','RowName',varname,'Units','normalized','Position',[0 0 1 1]);

% Correlation Matrix (Pearson)
R=corrcoef(data.X');
f2 = figure('Name', 'Correlation Matrix (in percentage)');
h=uitable(f2,'Data',abs(R*100),'ColumnName',varname,'RowName',varname,'Units','normalized','Position',[0 0 1 1]);

% High correlation but low chi-sq feature removal
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
data.dim=size(feat_2keep_index,1);
varname = varname(feat_2keep_index);
R=corrcoef(data.X');
f3 = figure('Name', 'Correlation Matrix after removing ambiguous variables (in percentage)');
h=uitable(f3,'Data',abs(R*100),'ColumnName',varname,'RowName',varname,'Units','normalized','Position',[0 0 1 1]);

%% REDUÇAO DE FEATURES A PARTIR DA AUC (utilizando a regra do trapezio para este calculo)

%passagem de classe 0e1 para 1e2
zeros=(find(data.y(1,:)==0));
ones=(find(data.y(1,:)==1));

aucdata.y(1,ones)=2;
aucdata.y(1,zeros)=1;

startup
%%
for i=1:size(data.X,1)
    
    [FP,FN]=roc(data.X(i,:),aucdata.y);
    
    TP=1-FN;
    AUC=trapz(FP,TP);
    figure; hold on; plot(FP,TP);
    title(['Feature ' int2str(i) ' com AUC de ' num2str(AUC)]);
    xlabel('false positives');
    ylabel('true positives');
    grid on;
end

%% PCA (comando matlab ao inves de usar STPRTOOLs)

% It computes Principal Component Analysis, i.e., the
%  linear transform which makes data uncorrelated and
%  minimize the reconstruction error.

[model,score,latent,tsquared,explained] = pca(data.X');


%% DIMENSION CHOICE

figure;
x=0:0.01:length(latent);
plot(1:length(latent),latent); hold on;
plot(x,ones(size(x)));
title('Scree plot');
xlabel('Num. Dimensões')
ylabel('Valores próprios')

%% CASO 1 (3 componentes)
[model_3,score_3,latent_3,~,explained_3] = pca(data.X', 'NumComponents',3);

data.X = score_3';
data.num_data = size(data.X,2);
data.dim = size(data.X,1);

%% labels ordenadas (classe 0 e 1)
types=[0 1];

allin=[];
labels=[];

for i=1:length(types)
    
    vin=find(data.y(1,:)==types(i)); %descobre valores de indices para os quais se verifica voltagem input

    % vetor com dados ordenados segundo a sua classe.
  
    allin=[allin data.X(:,vin)];
    labels=[labels data.y(1,vin)];
    
end


%% Definição de grupo treino e teste (escolha de igual numero de dados com diferente labelling) 

teste=[];
treino=[];
labels_teste=[];
labels_treino=[];

zeros.X = allin(:,1:6636);
zeros.y = labels(:,1:6636);
zeros.num_data = size(allin(:,1:6636),2);
zeros.dim = size(allin,1);
zeros.name = 'zero DATA';

ones.X = allin(:,23365:end);
ones.y = labels(:,23365:end);
ones.num_data = size(allin(:,23365:end),2);
ones.dim = size(allin,1);
ones.name = 'ones DATA';

treino=[zeros.X ones.X]';
labels_treino=[zeros.y ones.y];

teste=data.X';
labels_teste=data.y;

%% CLASSIFY WITH LDA

[stat_values_lda,lda_class,prob]=lda(treino,teste,labels_treino',labels_teste);
cerror(lda_class,labels_teste) %Classification error

%(scatter)plot CLUSTERS
%indexes

pay_ind=find(lda_class==1);
notpay_ind=find(lda_class==0);
figure('Name','Distribution of DATA according to labeling from LDA');
plot3(teste(pay_ind,1),teste(pay_ind,2),teste(pay_ind,3),'bo',teste(notpay_ind,1),teste(notpay_ind,2),teste(notpay_ind,3),'ro');
legend('able to pay','not able to pay');

%ROC CURVE
figure('name','ROC curve for LDA')

[x,y]=perfcurve(labels_teste,prob(:,2),1);
plot(x,y,'r-');grid on;
xlabel('FP rate (1-Specificity)'); ylabel('TP rate (Sensitivity)');
title('ROC curve for LDA')

C = cell(2,3);
C{1,1}='Pos'; %verdadeiros
C{2,1}='Neg'; %verdadeiros
C{1,2}=stat_values_lda(5);
C{1,3}=stat_values_lda(6);
C{2,2}=stat_values_lda(4);
C{2,3}=stat_values_lda(3);

cell2table(C,'VariableNames',{'LDA3' 'Pos' 'Neg'})
array2table([stat_values_lda(2),stat_values_lda(1),stat_values_lda(7)],'VariableNames',{'sensibilidade','especificidade','accuracy'})

%% CLASSIFY WITH DISTANCE FUNCTION

%Mahalanobis
[stat_values_mah,mah_class]= distfun(2,treino,teste,labels_treino',labels_teste' );
cerror(mah_class,labels_teste) %Classification error

%indexes
pay_ind=find(mah_class==1);
notpay_ind=find(mah_class==0);
figure('Name','Distribution of DATA according to labeling from Mahalanobis');
plot3(teste(pay_ind,1),teste(pay_ind,2),teste(pay_ind,3),'bo',teste(notpay_ind,1),teste(notpay_ind,2),teste(notpay_ind,3),'ro');
legend('able to pay','not able to pay');

C = cell(2,3);
C{1,1}='Pos'; %verdadeiros
C{2,1}='Neg'; %verdadeiros
C{1,2}=stat_values_mah(5);
C{1,3}=stat_values_mah(6);
C{2,2}=stat_values_mah(4);
C{2,3}=stat_values_mah(3);

cell2table(C,'VariableNames',{'MAH3' 'Pos' 'Neg'})
array2table([stat_values_mah(2),stat_values_mah(1),stat_values_mah(7)],'VariableNames',{'sensibilidade','especificidade','accuracy'})


%% Euclidean
[stat_values_euc,euc_class]= distfun(1,treino,teste,labels_treino',labels_teste );
cerror(euc_class,labels_teste) %Classification error

%(scatter)plot CLUSTERS
%indexes
pay_ind=find(euc_class==1);
notpay_ind=find(euc_class==0);
figure('Name','Distribution of DATA according to labeling from Euclidean distance');
plot3(teste(pay_ind,1),teste(pay_ind,2),teste(pay_ind,3),'bo',teste(notpay_ind,1),teste(notpay_ind,2),teste(notpay_ind,3),'ro');
legend('able to pay','not able to pay');

C = cell(2,3);
C{1,1}='Pos'; %verdadeiros
C{2,1}='Neg'; %verdadeiros
C{1,2}=stat_values_euc(5);
C{1,3}=stat_values_euc(6);
C{2,2}=stat_values_euc(4);
C{2,3}=stat_values_euc(3);

cell2table(C,'VariableNames',{'EUC3' 'Pos' 'Neg'})
array2table([stat_values_euc(2),stat_values_euc(1),stat_values_euc(7)],'VariableNames',{'sensibilidade','especificidade','accuracy'})

%% CLASSIFY WITH HIERARCHICAL CLUSTERING (PODE SE USAR PARA ESCOLHER VARIAVEIS?)

% eliminaçao de outlier na posiçao 4118
treino=[treino(1:4117,:);treino(4119:end,:)];
labels_treino=[labels_treino(1,1:4117) labels_treino(1,4119:end)];
%% Fazer parameterizaçao. Escolha dos parametros tem em conta a forma do scatterplot

[ stat_values_hier,clust_hier ] = hier( treino,2,labels_treino )
cerror(clust_hier,labels_treino) %Classification error

C = cell(2,3);
C{1,1}='Pos'; %verdadeiros
C{2,1}='Neg'; %verdadeiros
C{1,2}=stat_values_hier(5);
C{1,3}=stat_values_hier(6);
C{2,2}=stat_values_hier(4);
C{2,3}=stat_values_hier(3);

cell2table(C,'VariableNames',{'HIER3' 'Pos' 'Neg'})
array2table([stat_values_hier(2),stat_values_hier(1),stat_values_hier(7)],'VariableNames',{'sensibilidade','especificidade','accuracy'})

figure('Name','Distribution of DATA according to labeling from Hierarchical Clustering');
plot3(treino(find(clust_hier(:,1)==1),1),treino(find(clust_hier(:,1)==1),2),treino(find(clust_hier(:,1)==1),3),'bo')
hold on;
plot3(treino(find(clust_hier(:,1)==2),1),treino(find(clust_hier(:,1)==2),2),treino(find(clust_hier(:,1)==2),3),'ro')

%% STATISTICS CLASSIFICATION

% Verificação da gaussianidade para o metodo Bayesiano

for i=1:size(data.X,1)
    kolmogorov=kstest(data.X(i,:));
    fprintf(['Para a feature ' num2str(i) ' o kstest dá output de ' int2str(kolmogorov) '\n'])
end

%% labels ordenadas (classe 1 e 2)

zero_pos=(find(labels(1,:)==0));
one_pos=(find(labels(1,:)==1));

ordlabels(1,one_pos)=2;
ordlabels(1,zero_pos)=1;

cls_ver = ordlabels ;
%%
%  Description:
%  It computes Maximum Likelihood estimation of parameters
%  of Gaussian mixture model for given labeled data sample
%  (complete data).

%  A Gaussian mixture model is a probabilistic model that assumes all the data points are
%  generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

label1 = find(labels_treino==0);
label2 = find(labels_treino==1);
bayesmodel.Pclass{1} = mlcgmm(zeros.X);
bayesmodel.Pclass{2} = mlcgmm(ones.X);
bayesmodel.Prior = [length(label1) length(label2)]/(length(label1)+length(label2));
cls_test = bayescls(allin,bayesmodel);

cerror(cls_test,cls_ver)

%  This function implements the classifier minimizing the Bayesian risk
%  It corresponds to the minimization of
%  probability of misclassification. The input vectors X are classified
%  into classes with the highest a posterior probabilities computed from
%  given model.

%overfitting. K's muito altos.

%%

VP_lda=0;
FP_lda=0;
FN_lda=0;
VN_lda=0;

for i=1:length(cls_test)
    if( (isequal(cls_ver(i),2)==0) && (isequal(cls_test(i),2)==1))
        FP_lda=FP_lda+1;
    elseif((isequal(cls_ver(i),2)==1) && (isequal(cls_test(i),2)==0))
        FN_lda=FN_lda+1;
    elseif((isequal(cls_ver(i),2)==1) && (isequal(cls_test(i),2)==1))
        VP_lda=VP_lda+1;
    elseif((isequal(cls_ver(i),2)==0) && (isequal(cls_test(i),2)==0))
        VN_lda=VN_lda+1;
    end
    
end

specificity=VN_lda/(VN_lda+FP_lda);
sensitivity=VP_lda/(VP_lda+FN_lda);
accuracy=(VP_lda+VN_lda)/(VP_lda+FP_lda+FN_lda+VN_lda);
% FP_rate=FP_lda/(TN_lda+FP_lda);

stat_values_stat=[specificity,sensitivity,VN_lda,FP_lda,VP_lda,FN_lda,accuracy];

C = cell(2,3);
C{1,1}='Pos'; %verdadeiros
C{2,1}='Neg'; %verdadeiros
C{1,2}=stat_values_stat(5);
C{1,3}=stat_values_stat(6);
C{2,2}=stat_values_stat(4);
C{2,3}=stat_values_stat(3);

cell2table(C,'VariableNames',{'STAT3' 'Pos' 'Neg'})
array2table([stat_values_stat(2),stat_values_stat(1),stat_values_stat(7)],'VariableNames',{'sensibilidade','especificidade','accuracy'})
%%
cls_test=cls_test;
allin=allin';
figure('Name','Distribution of DATA according to labeling from Bayes Classification');
plot3(allin(find(cls_test(:,1)==1),1),allin(find(cls_test(:,1)==1),2),allin(find(cls_test(:,1)==1),3),'bo')
hold on;
plot3(allin(find(cls_test(:,1)==2),1),allin(find(cls_test(:,1)==2),2),allin(find(cls_test(:,1)==2),3),'ro')
legend('not able to pay','able to pay');

%% K-NN

down_x=downsample(teste,7);
down_label=downsample(labels_teste,7);
k=5;
%%
[stat_values_knn5,knn_class]=k_nn_loo(down_x,k,down_label);
cerror(knn_class,down_label) %Classification error

Algorithm={'k-nearest neighbour (k=5)'};
Sensitivity=[stat_values_knn5(2)];
Specificity=[stat_values_knn5(1)];
Accuracy=[stat_values_knn5(3)];
TABLE=table(Sensitivity,Specificity,Accuracy,'RowNames',Algorithm)

figure('Name','Distribution of DATA according to labeling from K-NN');
plot3(down_x(find(knn_class(:,1)==1),1),down_x(find(knn_class(:,1)==1),2),down_x(find(knn_class(:,1)==1),3),'bo')
hold on;
plot3(down_x(find(knn_class(:,1)==0),1),down_x(find(knn_class(:,1)==0),2),down_x(find(knn_class(:,1)==0),3),'ro')
legend('able to pay','not able to pay');

%% SVM (clustering): classificação supervisionada
ratio = 30;
X = score_3;
y = data.y';
ind_0=find(y==0);
ind_1=find(y==1);
sz = min(length(ind_0),length(ind_1));
ind_0=randsample(ind_0,round(sz*ratio/100));
ind_1=randsample(ind_1,round(sz*ratio/100));

Xtrain=X([ind_0;ind_1],:);
ytrain=y([ind_0;ind_1]);
X([ind_0;ind_1],:)=[];
y([ind_0;ind_1])=[];
Xtest=X;
ytest=y;

clearvars -except data Xtest ytest Xtrain ytrain varname


C = -1:3;
C = 10.^C;
gamma = C;
spec_rbf = zeros(length(C),length(gamma));
sens_rbf = zeros(length(C),length(gamma));
spec_lin = zeros(length(C),length(gamma));
sens_lin = zeros(length(C),length(gamma));
RBF_MODELS={};
LIN_MODELS={};
for i=1:length(C)
    for j=1:length(gamma)
        disp('rbf in')
        SVMModel_rbf = fitcsvm(Xtrain,ytrain,'BoxConstraint',C(i),'KernelFunction','rbf','KernelScale',gamma(j));
        disp('lin in')
        SVMModel_lin = fitcsvm(Xtrain,ytrain,'BoxConstraint',C(i),'KernelFunction','linear','KernelScale',gamma(j));
        
        newy_rbf = predict(SVMModel_rbf,Xtest);
        newy_lin = predict(SVMModel_lin,Xtest);
        C_rbf=confusionmat(newy_rbf,ytest);
        C_lin=confusionmat(newy_lin,ytest);
        
        sens_rbf(i,j) = C_rbf(2,2)/(C_rbf(2,2)+C_rbf(1,2));
        spec_rbf(i,j) = C_rbf(1,1)/(C_rbf(1,1)+C_rbf(2,1));
        sens_lin(i,j) = C_lin(2,2)/(C_lin(2,2)+C_lin(1,2));
        spec_lin(i,j) = C_lin(1,1)/(C_lin(1,1)+C_lin(2,1));
        
        disp('gamma')
        RBF_MODELS={RBF_MODELS{:},SVMModel_rbf};
        LIN_MODELS={LIN_MODELS{:},SVMModel_lin};
    end
    disp('c')
end
%%
VP_lda=0;
FP_lda=0;
FN_lda=0;
VN_lda=0;

for i=1:length(cls_test)
    if( (isequal(cls_ver(i),2)==0) && (isequal(cls_test(i),2)==1))
        FP_lda=FP_lda+1;
    elseif((isequal(cls_ver(i),2)==1) && (isequal(cls_test(i),2)==0))
        FN_lda=FN_lda+1;
    elseif((isequal(cls_ver(i),2)==1) && (isequal(cls_test(i),2)==1))
        VP_lda=VP_lda+1;
    elseif((isequal(cls_ver(i),2)==0) && (isequal(cls_test(i),2)==0))
        VN_lda=VN_lda+1;
    end
    
end

specificity=VN_lda/(VN_lda+FP_lda);
sensitivity=VP_lda/(VP_lda+FN_lda);
accuracy=(VP_lda+VN_lda)/(VP_lda+FP_lda+FN_lda+VN_lda);
% FP_rate=FP_lda/(TN_lda+FP_lda);

stat_values_svm=[specificity,sensitivity,VN_lda,FP_lda,VP_lda,FN_lda,accuracy];

C = cell(2,3);
C{1,1}='Pos'; %verdadeiros
C{2,1}='Neg'; %verdadeiros
C{1,2}=stat_values_svm(5);
C{1,3}=stat_values_svm(6);
C{2,2}=stat_values_svm(4);
C{2,3}=stat_values_svm(3);

cell2table(C,'VariableNames',{'STAT3' 'Pos' 'Neg'})
array2table([stat_values_svm(2),stat_values_svm(1),stat_values_svm(7)],'VariableNames',{'sensibilidade','especificidade','accuracy'})
%%

figure('Name','Distribution of DATA according to labeling from SVM');
plot3(allin(find(cls_test(:,1)==1),1),allin(find(cls_test(:,1)==1),2),allin(find(cls_test(:,1)==1),3),'bo')
hold on;
plot3(allin(find(cls_test(:,1)==2),1),allin(find(cls_test(:,1)==2),2),allin(find(cls_test(:,1)==2),3),'ro')
legend('not able to pay','able to pay');
%%
% Dados os resultados da 1a parte deste projecto, onde se verificou que a
% utilizaçao de 5 componentes acrescentava pouca informaçao à nossa
% analise, resolveu-se excluir essa parte, de forma a simplificar
% resultados.
