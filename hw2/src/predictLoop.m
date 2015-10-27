function pred= predictLoop(x,S,Y,X,indS,as, bw)
        for s=1:S
            pred(s)=(as(indS(s)).*Y(indS(s)).*exp(-1/(2*bw^2)*norm(X(indS(s), :)-x')^2));
            pred=sum(pred);
        end 
    end 