
gnuplot -persist << EOF
	set style data histogram
	set term pdf
	set output 'time$2.pdf'
	set grid
	set style histogram clustered gap 2
	set boxwidth 0.8
	set style fill solid 0.4 border
	set style fill pattern 4 border -1
	set ylabel 'Time'
	set key right top Left reverse width 0 box 3
	#set yrange[0:100]
	#set xtics('Conv1'0, 'Conv2'1)
	set xrange[0.5:12.5]
	set key font ', 10'
	set xtics font ', 10'
	set ytics font ', 10'
	plot '$1' using 2:xtic(1) title "im2col", '$1' using 3:xtic(1) title "direct", '$1' using 4:xtic(1) title "im2winBase", '$1' using 5:xtic(1) title "im2winMKL"
	quit
EOF

gnuplot -persist << EOF
	set style data histogram
	set term pdf
	set output 'GFLOPS$2.pdf'
	set grid
	set style histogram clustered gap 2
	set boxwidth 0.8
	set style fill solid 0.4 border
	set style fill pattern 4 border -1
	set ylabel 'GFLOPS'
	set key right top Left reverse width 0 box 3
	#set yrange[0:100]
	#set xtics('Conv1'0, 'Conv2'1)
	set xrange[0.5:12.5]
	set key font ', 10'
	set xtics font ', 10'
	set ytics font ', 10'
	plot '$1' using 6:xtic(1) title "im2col", '$1' using 7:xtic(1) title "direct", '$1' using 8:xtic(1) title "im2winBase", '$1' using 9:xtic(1) title "im2winMKL"
	quit
EOF
