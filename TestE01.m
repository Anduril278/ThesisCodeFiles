clc;
close all;
clear all;

if ispc
    d = '\';
else
    d = '/';
end

tl = 10;

path_bd = pwd;
sd = find(path_bd==d);
path_bd = [path_bd(1:sd(end-1)), 'senales', d, 'Erick', d];

fs = dir([path_bd, '*.txt']);
nfs = length(fs);

s1 = cell(1,nfs);
for i1=1:length(fs)
    s1{i1} = load([path_bd, fs(i1).name]);
end
nm = length(s1{1});

s = zeros(1,nfs*nm);
for i1=1:length(fs)
    s((i1-1)*nm+(1:nm)) = s1{i1};
end

dw_max = 71;
ntest = 30;
mAUC =zeros(ntest, (dw_max-1)/2, 3);
mAccuracy = zeros(ntest, (dw_max-1)/2,1);
mMC = cell(ntest, (dw_max-1)/2, 1);
for itest = 1:ntest
    for dw=3:2:dw_max
        fct_c = scn(s, dw)';
        clsc = [1*ones(nm,1); 2*ones(nm,1); 3*ones(nm,1)];
        idval = cvpartition(clsc,'HoldOut',0.5);
        dat_fct = fct_c(idval.training,:);
        dat_cls = clsc(idval.training);
        t1 = fitctree(dat_fct,dat_cls);
        t2 = prune(t1, 'Level', tl);
        clscp = predict(t1, fct_c(idval.test,:));
        clscp2 = predict(t2, fct_c(idval.test,:));
        clscv = clsc(idval.test);
        clscv2 = clsc(idval.test);
        
        % Calculo de AUC en la validacion
        cPr = (clscp==clscv);
        isMissing = isnan(clscp);
        cPr = cPr(~isMissing);
        clscv = clscv(~isMissing);
        validationAccuracy = sum(cPr)/length(clscv);

        cPr2 = (clscp2==clscv2);
        isMissing2 = isnan(clscp2);
        cPr2 = cPr2(~isMissing2);
        clscv2 = clscv2(~isMissing2);
        validationAccuracy2 = sum(cPr2)/length(clscv2);

        AUC = zeros(1,3);
        for i2=1:3
            clscvAUC = clscv==i2;
            clscpAUC = clscp==i2;
            [X1,Y1,T1,AUC(i2)] = perfcurve(clscvAUC,double(clscpAUC),'true');
            mAUC(itest, (dw-1)/2, i2) = AUC(i2);
        end
        mAccuracy(itest, (dw-1)/2) = validationAccuracy;
        % Calculo de la matriz de confucion en la validacion
        mMC{itest, (dw-1)/2} = zeros(3,3);
        for i2=1:3
            for i3=1:3
                mcr = (clscv==i2);
                mcp = (clscp==i3);
                mMC{itest, (dw-1)/2}(i2,i3) = sum(mcr&mcp);
            end
        end
    end
    fprintf('Prueba %d/%d\n', itest,ntest);
end

vdw = 3:2:dw_max;

figure(1)
plot(vdw, mean(mAccuracy));
grid on;

