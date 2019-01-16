% tc = 550 ¡Ó 32
% beta = 0:1/1024:1
% w = 0~492
% phi = 0:2£k/1024:2£k
function data = myLPPLGA(Y,n,sr,mr,g)
    % the first generation.
    population = rand(n, 40)>0.5;
    srnum = round(n*sr);
    mrnum = round(n*mr);
    data = [];
    for j = 1 : g
        disp(j);
        % calculate fitness
        for i = 1 : n
            tc = 550 + sum(population(i,1:6).*(2.^[0:5])) - 32;
            beta= sum(population(i,7:16).*(2.^[0:9]))./1024;
            w = sum(population(i,17:30).*(2.^[0:13]))*0.03;
            phi = (sum(population(i,31:40).*(2.^[0:9])))./1024.*(2*pi);
            reg = bubbleReg(tc,beta,w,phi);
            tempData = regress(log(Y(1:tc-1)),reg);
            E(i,1) = fitLPPL(Y,tempData(1),tempData(2),tempData(3)/tempData(2),tc,beta,w,phi);
        end
        E(:,2) = [1:n]'; %append index
        SE = sortrows(E,1); %used error to sort.
        % tournament selection
        population(1:srnum,:) = population(SE(1:srnum,2),:);
       
        % crossover used survival people to generate all people.
        for k = srnum+1 : n
            temp = randperm(srnum);
            father = population(temp(1),:);
            mother = population(temp(2),:);
            mask = rand(1,40)>0.5;
            son(mask==1) = father(mask==1);
            son(mask==0) = mother(mask==0);
            population(k,:)=son;
        end
        
        % mutation : select k size person to mutation one geneic.
        for k = 1 : mrnum
            person = ceil(rand(1)*n);
            geneno = ceil(rand(1)*40);
            population(person,geneno) =~ population(person,geneno);
        end
    end
    for i = 1 : n
        tc = 550 + sum(population(i,1:6).*(2.^[0:5])) - 32;
        beta= sum(population(i,7:16).*(2.^[0:9]))./1024;
        w = sum(population(i,17:30).*(2.^[0:13]))*0.03;
        phi = (sum(population(i,31:40).*(2.^[0:9])))./1024.*(2*pi);
        reg = bubbleReg(tc,beta,w,phi);
        tempData = regress(log(Y(1:tc-1)),reg);
        E(i,1) = fitLPPL(Y,tempData(1),tempData(2),tempData(3)/tempData(2),tc,beta,w,phi);
    end
    
    % select the best choice to return
    E(:,2) = [1:n]'; % append index
    SE = sortrows(E,1); % used error to sort.
    data.out = population(SE(1:srnum,2),:);

    data.tc = 550 + sum(data.out(1,1:6).*(2.^[0:5])) - 32;
    data.beta= sum(data.out(1,7:16).*(2.^[0:9]))/1024;
    data.w = sum(data.out(1,17:30).*(2.^[0:13]))*0.03;
    data.phi = (sum(data.out(1,31:40).*(2.^[0:9])))./1024.*(2*pi);
    reg = bubbleReg(tc,beta,w,phi);
    tempData = regress(log(Y(1:tc-1)),reg);
    data.A = tempData(1);
    data.B = tempData(2);
    data.C = tempData(3)/tempData(2);
    
end