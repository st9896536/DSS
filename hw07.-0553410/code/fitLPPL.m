function E = fitLPPL(Y,A,B,C,tc,beta,w,phi)
    E = sqrt(sum(exp(A + B*((tc-[1:tc-1]).^beta).*(1+C*cos(w*log(tc-[1:tc-1])+phi))) - Y(1:tc-1)').^2 /(tc-1));
end
