
gnuplot -persist << EOF
	set terminal pdfcairo size 20cm,8cm
	set style data histogram
	set term pdf
	set output 'Ablation$2.pdf'
	set grid ytics
	set style histogram clustered gap 1
	set boxwidth 0.8
	set style fill solid 0.4 border
	set style fill pattern 4 border -1
	set ylabel 'TFLOPS'
	set key right top Left reverse width 0 box 3
	#set yrange[0:100]
	#set xtics('Conv1'0, 'Conv2'1)
	set xrange[0.5:12.5]
	set font "Times New Roman,20,bold"
	set xtics rotate by 30 offset character -2,-1.2
	set style fill pattern 6
	plot '$1' using 11:xtic(1) title "im2winGPU",'$1' using 12:xtic(1) title "without prefetch",'$1' using 13:xtic(1) title "without vectorized load",'$1' using 14:xtic(1) title "without micro kernel"
	quit
EOF
