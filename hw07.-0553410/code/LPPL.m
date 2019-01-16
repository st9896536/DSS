function out = LPPL(A,B,C,tc,beta,w,phi)
    out = A + B*((tc-[1:tc-1]).^beta).*(1+C*cos(w*log(tc-[1:tc-1])+phi));
end