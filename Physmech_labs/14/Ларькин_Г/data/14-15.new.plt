reset session

set term wxt enhanced

flnm='run.csv'
set datafile separator  ";"

do for [i=0:31] {
	#eval("AI".gprintf('%g',i)."=1; print AI".gprintf('%g',i));
	eval("AI".gprintf('%g',i)."=1");
}
unused="\
AI20=0; AI21=0; AI22=0; AI23=0; AI24=0; AI25=0; AI26=0; AI27=0; AI28=0; AI29=0; AI30=0;\
"; eval unused

p1=747.0 #742 #[mmHg]
p2=99.5 #95 #[a.u.]


plotstr1="plot "
do for [i=0:31] {
	plotstr1=plotstr1."'".flnm."' every ::1::1 u (AI1_".gprintf('%g',i)."=$".gprintf('%g',i+1).", $0):".gprintf('%g',i+1).":".gprintf('%g',i+1+32)." w p t \"AI".gprintf('%g',i)."\", "
}

plotstr2="plot "
do for [i=0:31] {
	eval "k".gprintf('%g',i)."=1"
	eval "b".gprintf('%g',i)."=0"
	plotstr2=plotstr2."'".flnm."' every ::3::3 u (AI2_".gprintf('%g',i)."=$".gprintf('%g',i+1).", k".gprintf('%g',i)."=p2/100.0*9.81e4/(AI1_".gprintf('%g',i)."-AI2_".gprintf('%g',i)."), b".gprintf('%g',i)."=p1*133.33-k".gprintf('%g',i)."*AI1_".gprintf('%g',i).", $0):".gprintf('%g',i+1).":".gprintf('%g',i+1+32)." w p t \"AI".gprintf('%g',i)."\", "
}

checkp1p2="\
if (exists(\"p1\") && exists(\"p2\")) { \
	set title \"p1=\".gprintf('%g', p1).\" mmHg, p2=\".gprintf('%g', p2).\" a.u.\" tc rgb \"blue\" font \"Arial-Bold\"; \
	set yl \"p, Pa\"; \
	eval plotstr1; \
	eval plotstr2; \
} else { \
	print \"Enter p1 [mmHg] or/and p2 [a.u.]. Example: p1=745; p2=98.5\"; \
	set title \"Enter p1 [mmHg] or/and p2 [a.u.]. Example: \\\"p1=745; p2=98.5\\\". Then press F5\" tc rgb \"red\" font \"Arial-Bold\";\
	set yl \"U, V\"; \
}\
"
eval checkp1p2

set xl "t, s"

plotstr="plot "
do for [i=0:31] {
	plotstr=plotstr."'".flnm."' every ::5 u ((AI".gprintf('%g',i).">0) ? $0 : NaN ):($".gprintf('%g',i+1)."*k".gprintf('%g',i)."+b".gprintf('%g',i)."):($".gprintf('%g',i+1+32)."*k".gprintf('%g',i).") w yerrorlines t \"AI".gprintf('%g',i)."\", "
}

print plotstr
eval plotstr

hideAll="\
do for [i=0:31] { \
	eval(\"AI\".gprintf('%g',i).\"=0\"); \
}; \
"
showAll="\
do for [i=0:31] { \
	eval(\"AI\".gprintf('%g',i).\"=1\"); \
}; \
eval unused; \
eval plotstr; \
"
showNozzle="eval hideAll; \
AI0=1; AI1=1; AI2=1; AI3=1; AI4=1; AI5=1; AI6=1; AI7=1; AI31=1; \
eval plotstr; \
"

showPitotTubes="eval hideAll; \
AI8=1; AI9=1; AI10=1; AI11=1; AI12=1; AI13=1; AI14=1; AI15=1; AI16=1; AI17=1; AI18=1; AI19=1; AI7=1; AI31=1; \
eval plotstr; \
"


bind F5 "eval checkp1p2; eval plotstr"