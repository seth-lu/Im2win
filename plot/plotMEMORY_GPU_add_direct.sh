
gnuplot -persist << EOF
	set style data histogram
	set term pdf
	set output 'add_direct_MEMORY$2.pdf'
	set grid ytics
	set style histogram clustered gap 1
	set boxwidth 0.8
	set style fill solid 0.4 border
	set style fill pattern 4 border -1
	set ylabel 'Memory Usage (GB)'
	set key right top Left reverse width 0 box 3
	set yrange[0:]
	#set xtics('Conv1'0, 'Conv2'1)
	set xrange[0.5:12.5]
	set xtics font "arial,10"
	plot '$1' using 2:xtic(1) title "im2col+cuBLAS",'$1' using 3:xtic(1) title "cuDNN",'$1' using 6:xtic(1) title "im2winHPC",'$1' using 4:xtic(1) title "direct"
	quit
EOF
